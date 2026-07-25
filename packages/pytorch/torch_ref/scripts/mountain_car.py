"""DQN on MountainCar training script.

Aligned with `Example.MountainCar` (Idris).
Usage:
    python -m torch_ref.scripts.mountain_car [--lr 1e-3] [--epochs 1000] [--seed 42]
                                              [--shaping 10.0]
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.mountain_car import (
    NUM_ENVS,
    QNetwork,
    ReplayBuffer,
    dqn_episode_batched,
    dqn_update,
    eps_greedy_batched,
    evaluate,
    linear_epsilon,
    make_mountaincar_vec_env,
    obs_tensor,
)
from torch_ref.replay import write_replay
from torch_ref.scripts.dqn import RecordingRandom
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_elapsed, format_result, mem_suffix, set_device

if TYPE_CHECKING:
    import gymnasium as gym

# Idris registry name -> this script's parameter name, model-index prefixed
# (0 = online, 1 = target). Mirrors the entry in scripts/paired_examples.py,
# which check-step-oracle.py cross-checks.
PAIRED_PARAMS = {
    "online.linear_0.bias": "0.fc1.bias",
    "online.linear_0.weight": "0.fc1.weight",
    "online.linear_1.bias": "0.fc2.bias",
    "online.linear_1.weight": "0.fc2.weight",
    "online.linear_2.bias": "0.fc3.bias",
    "online.linear_2.weight": "0.fc3.weight",
    "target.linear_0.bias": "1.fc1.bias",
    "target.linear_0.weight": "1.fc1.weight",
    "target.linear_1.bias": "1.fc2.bias",
    "target.linear_1.weight": "1.fc2.weight",
    "target.linear_2.bias": "1.fc3.bias",
    "target.linear_2.weight": "1.fc3.weight",
}


def _internal_states(vec: gym.vector.SyncVectorEnv) -> np.ndarray:
    """The sub-envs' exact float64 states [N, 2] (position, velocity).

    The oracle episode feeds these to the networks and the buffer instead of
    Gymnasium's observations: the observation pipeline rounds through
    float32, idris-gym's does not, and that rounding would put a noise floor
    under the comparison well above what the replayed episode reaches."""
    return np.array(
        [np.asarray(e.unwrapped.state, dtype=np.float64) for e in cast("Any", vec).envs]
    )


def _reset_uniforms(states: np.ndarray) -> list[float]:
    """Invert MountainCar's start draw — position ~ U(-0.6, -0.4), velocity
    fixed at 0 — so each reset is ONE uniform: the one that produces the
    position under `Random.Dist.uniform` (`lo + u * (hi - lo)`)."""
    return [float(u) for u in (np.asarray(states).reshape(-1, 2)[:, 0] + 0.6) / 0.2]


def _oracle_episode(
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    vec: gym.vector.SyncVectorEnv,
    rec: RecordingRandom,
    batch_size: int,
    gamma: float,
    target_sync_every: int,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    shaping: float,
) -> list[float]:
    """`dqn_episode_batched`'s oracle twin: exact-state obs, shaped-reward
    pushes, every reset draw logged as the uniform that produced it (initial
    and same-step auto-reset, env-ascending), the real `eps_greedy_batched`
    and `dqn_update` drawing through the recording rng. Returns the logged
    env uniforms; the gate/action/index draws land on `rec`."""
    states = _internal_states(vec)
    env_uniforms = _reset_uniforms(states)
    step_count = 0
    while True:
        obs_t = obs_tensor(states)  # [N, 2]
        epsilon = linear_epsilon(step_count, eps_start, eps_end, eps_decay)
        actions_np = eps_greedy_batched(q, obs_t, epsilon, rec)
        # SyncVectorEnv.step's stub returns unsolved TypeVars (Unknown).
        _, raw_rewards, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec.step(actions_np),
        )
        dones_np = np.logical_or(terms_np, truncs_np)
        new_states = _internal_states(vec)
        for i in np.flatnonzero(dones_np):
            env_uniforms += _reset_uniforms(new_states[int(i)])
        for i in range(NUM_ENVS):
            # Both sides shape with the post-step velocity — for a done env
            # that is the fresh reset state's (0.0), same-step autoreset.
            shaped_r = float(raw_rewards[i]) + shaping * abs(float(new_states[i, 1]))
            buffer.push(
                states[i].tolist(),
                int(actions_np[i]),
                shaped_r,
                new_states[i].tolist(),
                bool(dones_np[i]),
            )
        states = new_states
        step_count += 1
        if len(buffer) >= batch_size:
            dqn_update(q, target, optimizer, buffer, batch_size, gamma, rec)
        if step_count % target_sync_every == 0:
            target.load_state_dict(q.state_dict())
        if dones_np[0]:
            return env_uniforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000, help="number of episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer-cap", dest="buffer", type=int, default=50000)
    parser.add_argument("--target-sync", type=int, default=200)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=50000)
    parser.add_argument("--shaping", type=float, default=10.0)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for tensor ops (default: cpu)",
    )
    args = parser.parse_args()

    set_device(args.device)
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
    rng = random.Random(args.seed)

    print("=== DQN on MountainCar ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} gamma={args.gamma}"
        f" batch={args.batch} buffer={args.buffer}"
        f" target_sync={args.target_sync}"
        f" eps={args.eps_start}->{args.eps_end} shaping={args.shaping}"
        f" seed={args.seed}"
    )

    q = QNetwork().to(args.device)
    target = copy.deepcopy(q)
    maybe_dump_init(q, target)
    optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    # Oracle run: publish the fixture this side started from — both nets'
    # parameters, plus every draw one full episode makes, recorded to a
    # replay file — then run exactly that episode (collection interleaved
    # with replay updates) and publish the post-episode parameters. Same
    # protocol as dqn.py's, with MountainCar's one-uniform resets and
    # shaped-reward pushes; the episode always ends by the 200-step
    # truncation, so the truncation done-flag semantics are under test too.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        vec0, _obs0 = make_mountaincar_vec_env(args.seed, NUM_ENVS)
        maybe_dump_oracle((q, target), PAIRED_PARAMS)
        rec = RecordingRandom(args.seed)
        env_uniforms = _oracle_episode(
            q,
            target,
            optimizer,
            buffer,
            vec0,
            rec,
            args.batch,
            args.gamma,
            args.target_sync,
            args.eps_start,
            args.eps_end,
            args.eps_decay,
            args.shaping,
        )
        write_replay(
            os.environ["IDRISML_ORACLE_DUMP"] + ".replay",
            env=env_uniforms,
            choices=rec.decisions,
            uniforms=rec.uniforms,
        )
        maybe_dump_after_step((q, target), PAIRED_PARAMS)

    vec_env, obs0 = make_mountaincar_vec_env(args.seed, NUM_ENVS)
    obs_state = [obs0]
    step_count = [0]
    print()

    def epoch_fn() -> float:
        new_step, ep_return, new_obs = dqn_episode_batched(
            vec_env,
            obs_state[0],
            q,
            target,
            optimizer,
            buffer,
            step_count[0],
            args.batch,
            args.gamma,
            args.target_sync,
            args.eps_start,
            args.eps_end,
            args.eps_decay,
            args.shaping,
            rng,
        )
        step_count[0] = new_step
        obs_state[0] = new_obs
        return -ep_return

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=30), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t_start = time.monotonic()
    history: list[float] = []
    for ep in range(args.epochs):
        loss_val = epoch_fn()
        ep_return = -loss_val
        history.append(ep_return)
        if (ep + 1) % 50 == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {ep + 1}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={ep_return:.1f}\trecent_50={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} episodes, {ms_per_ep:.0f}ms/episode)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    avg = evaluate(q)
    print(f"Eval (30 episodes, greedy): avg_return={avg:.1f}")
    print()
    print(
        format_result(
            [
                ("avg_return", f"{avg:.1f}"),
                ("epochs", str(args.epochs)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

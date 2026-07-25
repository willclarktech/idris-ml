"""DQN on CartPole training script with --lr-find support.

Output format and epoch semantics align with `Example.Dqn` (Idris):
each "epoch" = one full episode + intra-episode replay updates, and
the "loss" reported to lr_find is `-episode_return` (matching the
Idris convention so the cross-backend lr_find comparison is meaningful).

Usage:
    python -m torch_ref.scripts.dqn [--lr 5e-4] [--epochs 300] [--seed 42]
                                     [--lr-find]
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
from torch_ref.models.dqn import (
    NUM_ENVS,
    QNetwork,
    ReplayBuffer,
    dqn_episode_batched,
    dqn_update,
    eps_greedy_batched,
    evaluate,
    linear_epsilon,
    make_cartpole_vec_env,
)
from torch_ref.models.reinforce import obs_tensor
from torch_ref.replay import write_replay
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_elapsed, format_result, mem_suffix, set_device

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym

    # (q, target, optimizer, buffer, batch_size, gamma, rng) -> loss
    UpdateFn = Callable[
        [QNetwork, QNetwork, torch.optim.Optimizer, ReplayBuffer, int, float, random.Random],
        float,
    ]

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


class RecordingRandom(random.Random):
    """`random.Random` that also logs every draw the epoch makes, in call
    order: `random()` gates onto the uniform stream, `randrange()` outcomes
    (explored actions and minibatch indices alike) onto the decision stream.
    `eps_greedy_batched` and `dqn_update` run unmodified with this as their
    rng, and the two streams are exactly the Idris side's uniform and choice
    channels in its consumption order."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.uniforms: list[float] = []
        self.decisions: list[int] = []

    def random(self) -> float:
        u = super().random()
        self.uniforms.append(u)
        return u

    def getrandbits(self, k: int) -> int:
        # Overriding random() alone flips CPython's Random onto its
        # random()-based _randbelow, making every randrange() draw (and log)
        # a uniform too. Redeclaring getrandbits keeps the bits-based path,
        # so the uniform stream stays gates-only.
        return super().getrandbits(k)

    def randrange(self, start: Any, stop: Any = None, step: Any = 1) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        n = super().randrange(start, stop, step)
        self.decisions.append(int(n))
        return int(n)


def _internal_states(vec: gym.vector.SyncVectorEnv) -> np.ndarray:
    """The sub-envs' exact float64 states, [N, 4].

    The oracle episode feeds these to the networks and the buffer instead of
    Gymnasium's observations: the observation pipeline rounds through
    float32, idris-gym's does not, and that rounding would put a noise floor
    under the comparison well above what the replayed episode reaches."""
    return np.array(
        [np.asarray(e.unwrapped.state, dtype=np.float64) for e in cast("Any", vec).envs]
    )


def _reset_uniforms(states: np.ndarray) -> list[float]:
    """Invert CartPole's U(-0.05, 0.05) start draw: the uniform that produces
    `state` under `Random.Dist.uniform` (`lo + u * (hi - lo)`). Exact up to
    one rounding of the division."""
    return [float(u) for u in (np.asarray(states).reshape(-1) + 0.05) / 0.1]


def oracle_episode(
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    vec: gym.vector.SyncVectorEnv,
    rec: RecordingRandom,
    batch_size: int,
    gamma: float,
    target_sync_every: int,
    update_fn: UpdateFn = dqn_update,
) -> list[float]:
    """`dqn_episode_batched`'s oracle twin: exact-state obs, every reset draw
    logged as the uniforms that produced it (initial and same-step
    auto-reset, env-ascending), the real `eps_greedy_batched` and
    `update_fn` (`dqn_update`; double_dqn's script passes its own) drawing
    through the recording rng. Returns the logged env uniforms; the
    action/gate/index draws land on `rec`."""
    states = _internal_states(vec)
    env_uniforms = _reset_uniforms(states)
    step_count = 0
    while True:
        obs_t = obs_tensor(states)  # [N, 4]
        epsilon = linear_epsilon(step_count)
        actions_np = eps_greedy_batched(q, obs_t, epsilon, rec)
        # SyncVectorEnv.step's stub returns unsolved TypeVars (Unknown).
        _, rewards_np, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec.step(actions_np),
        )
        dones_np = np.logical_or(terms_np, truncs_np)
        new_states = _internal_states(vec)
        for i in np.flatnonzero(dones_np):
            env_uniforms += _reset_uniforms(new_states[int(i)])
        for i in range(NUM_ENVS):
            # For a done env, `new_states[i]` is the fresh reset draw; the
            # transition's next_obs is that restarted state on both sides
            # (same-step autoreset).
            buffer.push(
                states[i].tolist(),
                int(actions_np[i]),
                float(rewards_np[i]),
                new_states[i].tolist(),
                bool(dones_np[i]),
            )
        states = new_states
        step_count += 1
        if len(buffer) >= batch_size:
            update_fn(q, target, optimizer, buffer, batch_size, gamma, rec)
        if step_count % target_sync_every == 0:
            target.load_state_dict(q.state_dict())
        if dones_np[0]:
            return env_uniforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300, help="number of episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer-cap", dest="buffer", type=int, default=10000)
    parser.add_argument("--target-sync", type=int, default=100)
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

    print("=== DQN on CartPole ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} gamma={args.gamma}"
        f" batch={args.batch} buffer={args.buffer}"
        f" target_sync={args.target_sync} seed={args.seed}"
    )

    q = QNetwork().to(args.device)
    target = copy.deepcopy(q)
    maybe_dump_init(q, target)
    optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    # Oracle run: publish the fixture this side started from — both nets'
    # parameters, plus every draw one full episode makes, recorded to a
    # replay file — then run exactly that episode (collection interleaved
    # with replay updates, as always) and publish the post-episode
    # parameters. The Idris side replays the draws through `--replay`:
    # explore gates land on its uniform channel, explored actions and
    # minibatch indices on its decision channel, reset states on its env
    # `Source` (as the uniforms that produced them), so its own physics and
    # forward passes regenerate the identical episode, and the TD targets,
    # the mse loss, the clip and Adam are all under test.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        vec0, _obs0 = make_cartpole_vec_env(args.seed, NUM_ENVS)
        maybe_dump_oracle((q, target), PAIRED_PARAMS)
        rec = RecordingRandom(args.seed)
        env_uniforms = oracle_episode(
            q, target, optimizer, buffer, vec0, rec, args.batch, args.gamma, args.target_sync
        )
        write_replay(
            os.environ["IDRISML_ORACLE_DUMP"] + ".replay",
            env=env_uniforms,
            choices=rec.decisions,
            uniforms=rec.uniforms,
        )
        maybe_dump_after_step((q, target), PAIRED_PARAMS)

    vec_env, obs0 = make_cartpole_vec_env(args.seed, NUM_ENVS)
    obs_state = [obs0]
    step_count = [0]
    print()

    def epoch_fn() -> float:
        """One batched DQN episode. Returns -env-0_episode_return
        (matches Idris loss)."""
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
            rng,
        )
        step_count[0] = new_step
        obs_state[0] = new_obs
        return -ep_return  # Idris reports `negate ret`

    if args.lr_find:
        # 30 iters: each iter is a full episode (up to 200 steps), heavier
        # than supervised so we use the same count as Idris.
        lr_find(LrFindConfig(num_iters=30), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t_start = time.monotonic()
    history: list[float] = []
    for epoch in range(args.epochs):
        loss_val = epoch_fn()
        ep_return = -loss_val
        history.append(ep_return)
        if (epoch + 1) % 50 == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={ep_return:.1f}\trecent_50={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} episodes, {ms_per_ep:.0f}ms/episode)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    print("Eval (50 episodes, greedy):")
    avg_return = evaluate(q, n_episodes=50)
    print(f"  avg_return={avg_return:.1f}")

    print()
    print(
        format_result(
            [
                ("avg_return", f"{avg_return:.1f}"),
                ("epochs", str(args.epochs)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

"""SAC on MountainCarContinuous training script.

Aligned with `Example.MountainCarCont` (Idris).
Usage:
    python -m torch_ref.scripts.mountain_car_cont [--lr 3e-4] [--epochs 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.mountain_car_cont import (
    MAX_ACTION,
    MAX_STEPS,
    NUM_ENVS,
    Actor,
    QNet,
    ReplayBuffer,
    evaluate,
    make_mountaincarcont_vec_env,
    obs_tensor,
    polyak_update,
    sac_update,
)
from torch_ref.replay import write_replay
from torch_ref.scripts.dqn import RecordingRandom
from torch_ref.training.runner import (
    format_elapsed,
    format_result,
    get_device,
    mem_suffix,
    set_device,
)

# Idris registry name -> this script's parameter name, model-index prefixed
# (0 = actor, 1 = q1, 2 = q1_target, 3 = q2, 4 = q2_target). Mirrors the
# entry in scripts/paired_examples.py, which check-step-oracle.py
# cross-checks.
PAIRED_PARAMS = {
    "actor_.linear_0.bias": "0.fc1.bias",
    "actor_.linear_0.weight": "0.fc1.weight",
    "actor_.linear_1.bias": "0.fc2.bias",
    "actor_.linear_1.weight": "0.fc2.weight",
    "actor_.linear_2.bias": "0.mean_head.bias",
    "actor_.linear_2.weight": "0.mean_head.weight",
    "actor_log_std": "0.log_std",
    "q1_.linear_0.bias": "1.fc1.bias",
    "q1_.linear_0.weight": "1.fc1.weight",
    "q1_.linear_1.bias": "1.fc2.bias",
    "q1_.linear_1.weight": "1.fc2.weight",
    "q1_.linear_2.bias": "1.head.bias",
    "q1_.linear_2.weight": "1.head.weight",
    "q1tgt_.linear_0.bias": "2.fc1.bias",
    "q1tgt_.linear_0.weight": "2.fc1.weight",
    "q1tgt_.linear_1.bias": "2.fc2.bias",
    "q1tgt_.linear_1.weight": "2.fc2.weight",
    "q1tgt_.linear_2.bias": "2.head.bias",
    "q1tgt_.linear_2.weight": "2.head.weight",
    "q2_.linear_0.bias": "3.fc1.bias",
    "q2_.linear_0.weight": "3.fc1.weight",
    "q2_.linear_1.bias": "3.fc2.bias",
    "q2_.linear_1.weight": "3.fc2.weight",
    "q2_.linear_2.bias": "3.head.bias",
    "q2_.linear_2.weight": "3.head.weight",
    "q2tgt_.linear_0.bias": "4.fc1.bias",
    "q2tgt_.linear_0.weight": "4.fc1.weight",
    "q2tgt_.linear_1.bias": "4.fc2.bias",
    "q2tgt_.linear_1.weight": "4.fc2.weight",
    "q2tgt_.linear_2.bias": "4.head.bias",
    "q2tgt_.linear_2.weight": "4.head.weight",
}


def _internal_state(env: gym.Env[np.ndarray, np.ndarray]) -> np.ndarray:
    """The sub-env's exact float64 state (position, velocity) — which for
    MountainCarContinuous is also the observation."""
    return np.asarray(cast("Any", env).unwrapped.state, dtype=np.float64)


def _reset_uniforms(state: np.ndarray) -> list[float]:
    """Invert MountainCarContinuous' start draw — position ~ U(-0.6, -0.4),
    velocity fixed at 0 — so each reset is ONE uniform under
    `Random.Dist.uniform` (`lo + u * (hi - lo)`)."""
    return [(float(state[0]) + 0.6) / 0.2]


def _oracle_step(args: argparse.Namespace) -> None:
    """One full SAC step under the oracle, sac.py's protocol with
    MountainCarContinuous' one-uniform resets and shaped-reward pushes:
    publish the fixture, run one collect step + (config allowing) one
    sac_update + polyaks, publish the post-step parameters. Envs are
    stepped UNWRAPPED with float64 actions (the vec wrapper casts to
    float32); the buffer done flag is TRUE termination only, exactly as
    the training loops on both sides push it."""
    rec = RecordingRandom(args.seed)
    actor = Actor().to(get_device())
    q1 = QNet().to(get_device())
    q2 = QNet().to(get_device())
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=args.lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)
    models = (actor, q1, q1_target, q2, q2_target)
    maybe_dump_oracle(models, PAIRED_PARAMS)

    envs: list[gym.Env[np.ndarray, np.ndarray]] = []
    env_uniforms: list[float] = []
    for i in range(NUM_ENVS):
        env = cast(
            "gym.Env[np.ndarray, np.ndarray]",
            gym.make("MountainCarContinuous-v0"),  # pyright: ignore[reportUnknownMemberType]
        )
        env.reset(seed=args.seed + i)
        envs.append(env)
        env_uniforms += _reset_uniforms(_internal_state(env))
    states = [_internal_state(e) for e in envs]

    if args.warmup > 0:
        actions = [rec.uniform(-MAX_ACTION, MAX_ACTION) for _ in range(NUM_ENVS)]
    else:
        obs_t = obs_tensor(np.array(states))
        with torch.no_grad():
            a_t, _ = actor.sample(obs_t, rec)
        # torch stub: Tensor.tolist() returns list[Unknown].
        actions = [float(a) for a in cast("list[float]", a_t.tolist())]  # pyright: ignore[reportUnknownMemberType]

    for i, env in enumerate(envs):
        step_out = cast("Any", env).unwrapped.step(np.array([actions[i]], dtype=np.float64))
        raw_reward = float(cast("float", step_out[1]))
        terminated = bool(cast("bool", step_out[2]))
        new_state = _internal_state(env)
        # ep_len goes 0 -> 1 in this single step; MAX_STEPS is 999, so no
        # env caps out and no reset draw is logged.
        assert MAX_STEPS > 1
        shaped_r = raw_reward + args.shaping * abs(float(new_state[1]))
        buffer.push(
            [float(x) for x in states[i]],
            float(actions[i]),
            shaped_r,
            [float(x) for x in new_state],
            terminated,
        )

    if len(buffer) >= max(args.batch, args.warmup):
        sac_update(
            actor,
            q1,
            q2,
            q1_target,
            q2_target,
            actor_opt,
            q1_opt,
            q2_opt,
            buffer,
            args.batch,
            args.gamma,
            args.alpha,
            rec,
        )
        polyak_update(q1_target, q1, args.tau)
        polyak_update(q2_target, q2, args.tau)

    write_replay(
        os.environ["IDRISML_ORACLE_DUMP"] + ".replay",
        env=env_uniforms,
        choices=rec.decisions,
        uniforms=rec.uniforms,
        normals=rec.normals,
    )
    maybe_dump_after_step(models, PAIRED_PARAMS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30000, help="number of env steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer-cap", dest="buffer", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--shaping", type=float, default=10.0)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Stub for API consistency; SAC's per-step + warmup don't fit lr_find.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for tensor ops (default: cpu)",
    )
    args = parser.parse_args()

    set_device(args.device)

    if args.lr_find:
        print("lr_find skipped for SAC: per-step epochs + warmup don't fit")
        print("the LR-range-test pattern. See docs/develop/hyperparameter-tuning-2026.md.")
        sys.exit(0)

    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
    rng = random.Random(args.seed)

    # Oracle run: one collect step + one update from a warmup-free config
    # (check-step-oracle.py passes --warmup 0 --batch 4 to both sides).
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        _oracle_step(args)

    print("=== SAC on MountainCarContinuous ===")
    print(
        f"Config: lr={args.lr} steps={args.epochs} gamma={args.gamma}"
        f" alpha={args.alpha} batch={args.batch} buffer={args.buffer}"
        f" warmup={args.warmup} tau={args.tau} shaping={args.shaping}"
        f" seed={args.seed}"
    )

    actor = Actor().to(args.device)
    q1 = QNet().to(args.device)
    q2 = QNet().to(args.device)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    maybe_dump_init(actor, q1, q2, q1_target, q2_target)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=args.lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    print()
    history: list[float] = []
    vec_env, obs_np = make_mountaincarcont_vec_env(args.seed, NUM_ENVS)
    ep_returns_running = np.zeros(NUM_ENVS, dtype=np.float64)
    t_start = time.monotonic()
    for step in range(args.epochs):
        if step < args.warmup:
            actions_np = np.array(
                [rng.uniform(-MAX_ACTION, MAX_ACTION) for _ in range(NUM_ENVS)],
                dtype=np.float64,
            )
        else:
            obs_t = obs_tensor(obs_np)
            with torch.no_grad():
                a_t, _ = actor.sample(obs_t, rng)
                actions_np = a_t.cpu().numpy().astype(np.float64)
        # SyncVectorEnv is unparameterized upstream; narrow its step() products.
        next_obs_np, raw_rewards, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec_env.step(actions_np.astype(np.float32).reshape(NUM_ENVS, 1)),
        )
        next_obs_np = next_obs_np.astype(np.float64)
        is_dones = np.logical_or(terms_np, truncs_np)
        ep_returns_running += raw_rewards.astype(np.float64)
        for i in range(NUM_ENVS):
            shaped_r = float(raw_rewards[i]) + args.shaping * abs(float(next_obs_np[i, 1]))
            buffer.push(
                obs_np[i].tolist(),
                float(actions_np[i]),
                shaped_r,
                next_obs_np[i].tolist(),
                bool(terms_np[i]),
            )
        for i in range(NUM_ENVS):
            if is_dones[i]:
                history.append(float(ep_returns_running[i]))
                ep_returns_running[i] = 0.0
                # SyncVectorEnv (SAME_STEP) already auto-reset this sub-env
                # through its own randomized start distribution and returned
                # the fresh obs in next_obs_np[i]. Until 2026-08-03 this loop
                # overwrote that with the pinned (-0.5, 0) center reset — the
                # remnant this file missed in the 2026-08-01 reset alignment.
        obs_np = next_obs_np
        if len(buffer) >= max(args.batch, args.warmup):
            sac_update(
                actor,
                q1,
                q2,
                q1_target,
                q2_target,
                actor_opt,
                q1_opt,
                q2_opt,
                buffer,
                args.batch,
                args.gamma,
                args.alpha,
                rng,
            )
            polyak_update(q1_target, q1, args.tau)
            polyak_update(q2_target, q2, args.tau)
        if (step + 1) % 2000 == 0:
            recent = history[-20:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {step + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_20={sum(recent) / len(recent):.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} steps, {ms_per_ep:.2f}ms/step)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    avg = evaluate(actor)
    print(f"Eval (20 episodes, greedy): avg_return={avg:.1f}")
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

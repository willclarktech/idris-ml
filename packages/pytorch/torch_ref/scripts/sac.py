"""SAC on Pendulum training script.

Aligned with `Example.Sac` (Idris).
Usage:
    python -m torch_ref.scripts.sac [--lr 3e-4] [--epochs 30000] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from torch_ref.init_manifest import maybe_dump_after_step, maybe_dump_oracle
from torch_ref.models.sac import (
    MAX_ACTION,
    MAX_STEPS,
    NUM_ENVS,
    Actor,
    QNet,
    ReplayBuffer,
    evaluate,
    obs_tensor,
    polyak_update,
    sac_update,
    train_sac,
)
from torch_ref.replay import write_replay
from torch_ref.scripts.dqn import RecordingRandom
from torch_ref.training.runner import format_result, get_device, set_device

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
    """The sub-env's exact float64 state (theta, theta_dot)."""
    return np.asarray(cast("Any", env).unwrapped.state, dtype=np.float64)


def _reset_uniforms(state: np.ndarray) -> list[float]:
    """Invert Pendulum's start draw — theta ~ U(-pi, pi) then theta_dot ~
    U(-1, 1) — into the uniforms that produce it under
    `Random.Dist.uniform` (`lo + u * (hi - lo)`)."""
    th, dth = float(state[0]), float(state[1])
    return [(th + math.pi) / (2.0 * math.pi), (dth + 1.0) / 2.0]


def _obs3(state: np.ndarray) -> list[float]:
    """[cos(theta), sin(theta), theta_dot] from the exact state — the
    `Gym.ClassicControl.Pendulum.pObserve` transform, float64 throughout."""
    th, dth = float(state[0]), float(state[1])
    return [math.cos(th), math.sin(th), dth]


def _oracle_step(args: argparse.Namespace) -> None:
    """One full SAC step under the oracle: publish the fixture (all five
    nets' parameters plus every draw the step makes), run exactly one
    collect step + (if the config allows) one sac_update + polyaks, publish
    the post-step parameters.

    The envs are stepped UNWRAPPED with float64 actions: the vec wrapper
    casts actions to the Box dtype (float32), and that rounding would move
    the trajectory off the Idris side's full-precision physics. Episode
    caps are per-env counters exactly as `Example.Sac.stepAllAutoResetP`
    keeps them (Pendulum never terminates on its own)."""
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
        env = cast("gym.Env[np.ndarray, np.ndarray]", gym.make("Pendulum-v1"))  # pyright: ignore[reportUnknownMemberType]
        env.reset(seed=args.seed + i)
        envs.append(env)
        env_uniforms += _reset_uniforms(_internal_state(env))
    states = [_internal_state(e) for e in envs]

    if args.warmup > 0:
        actions = [rec.uniform(-MAX_ACTION, MAX_ACTION) for _ in range(NUM_ENVS)]
    else:
        obs_t = obs_tensor(np.array([_obs3(s) for s in states]))
        with torch.no_grad():
            a_t, _ = actor.sample(obs_t, rec)
        # torch stub: Tensor.tolist() returns list[Unknown].
        actions = [float(a) for a in cast("list[float]", a_t.tolist())]  # pyright: ignore[reportUnknownMemberType]

    for i, env in enumerate(envs):
        step_out = cast("Any", env).unwrapped.step(np.array([actions[i]], dtype=np.float64))
        reward = float(cast("float", step_out[1]))
        new_state = _internal_state(env)
        # ep_len goes 0 -> 1 in this single step; MAX_STEPS is 200, so no
        # env caps out and no reset draw is logged.
        done = MAX_STEPS <= 1
        buffer.push(_obs3(states[i]), float(actions[i]), reward, _obs3(new_state), done)

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

    # torch stub: manual_seed's seed param is unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]

    # Oracle run: one collect step + one update from a warmup-free config
    # (check-step-oracle.py passes --warmup 0 --batch 4 to both sides), so
    # the replayed comparison covers the actor sample, the shared TD target,
    # both Q losses, the reparameterized actor loss and the polyak syncs.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        _oracle_step(args)

    print("=== SAC on Pendulum ===")
    print(
        f"Config: lr={args.lr} steps={args.epochs} gamma={args.gamma}"
        f" alpha={args.alpha} batch={args.batch} buffer={args.buffer}"
        f" warmup={args.warmup} tau={args.tau} seed={args.seed}"
    )
    print()

    t_start = time.monotonic()
    actor, _history = train_sac(
        total_steps=args.epochs,
        buffer_capacity=args.buffer,
        batch_size=args.batch,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        warmup_steps=args.warmup,
        tau=args.tau,
        seed=args.seed,
    )
    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} steps, {ms_per_ep:.2f}ms/step)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    avg = evaluate(actor, n_episodes=20)
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

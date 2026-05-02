"""SAC on MountainCarContinuous training script.

Aligned with `Example.MountainCarCont` (Idris).
Usage:
    python -m torch_ref.scripts.mountain_car_cont [--lr 3e-4] [--epochs 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time

import numpy as np
import torch

from torch_ref.models.mountain_car_cont import (
    MAX_ACTION,
    Actor,
    QNet,
    ReplayBuffer,
    evaluate,
    make_mountaincarcont_env,
    obs_tensor,
    polyak_update,
    reset_to_center,
    sac_update,
)
from torch_ref.training.runner import format_elapsed, format_result, mem_suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30000,
                        help="number of env steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer-cap", dest="buffer", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--shaping", type=float, default=10.0)
    parser.add_argument(
        "--lr-find", action="store_true",
        help="Stub for API consistency; SAC's per-step + warmup don't fit lr_find.",
    )
    args = parser.parse_args()

    if args.lr_find:
        print("lr_find skipped for SAC: per-step epochs + warmup don't fit")
        print("the LR-range-test pattern. See docs/develop/hyperparameter-tuning-2026.md.")
        sys.exit(0)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("=== SAC on MountainCarContinuous ===")
    print(
        f"Config: lr={args.lr} steps={args.epochs} gamma={args.gamma}"
        f" alpha={args.alpha} batch={args.batch} buffer={args.buffer}"
        f" warmup={args.warmup} tau={args.tau} shaping={args.shaping}"
        f" seed={args.seed}"
    )

    actor = Actor()
    q1 = QNet()
    q2 = QNet()
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=args.lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    print()
    history: list[float] = []
    env = make_mountaincarcont_env(args.seed)
    obs_np = reset_to_center(env)
    ep_return = 0.0
    t_start = time.monotonic()
    for step in range(args.epochs):
        obs = obs_tensor(obs_np)
        if step < args.warmup:
            action = rng.uniform(-MAX_ACTION, MAX_ACTION)
        else:
            with torch.no_grad():
                a_t, _ = actor.sample(obs)
                action = float(a_t.item())
        next_obs_np, raw_reward, terminated, truncated, _ = env.step(
            np.array([action], dtype=np.float32)
        )
        next_obs_np = next_obs_np.astype(np.float64)
        ep_return += float(raw_reward)
        buffer_done = bool(terminated)
        shaped = float(raw_reward) + args.shaping * abs(float(next_obs_np[1]))
        buffer.push(obs_np.tolist(), action, shaped, next_obs_np.tolist(), buffer_done)
        is_done = bool(terminated or truncated)
        obs_np = next_obs_np
        if is_done:
            history.append(ep_return)
            ep_return = 0.0
            env.reset()
            obs_np = reset_to_center(env)
        if len(buffer) >= max(args.batch, args.warmup):
            sac_update(
                actor, q1, q2, q1_target, q2_target,
                actor_opt, q1_opt, q2_opt, buffer, args.batch,
                args.gamma, args.alpha, rng,
            )
            polyak_update(q1_target, q1, args.tau)
            polyak_update(q2_target, q2, args.tau)
        if (step + 1) % 2000 == 0:
            recent = history[-20:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {step + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_20={sum(recent)/len(recent):.1f}"
            )

    print()
    avg = evaluate(actor)
    print(f"Eval (20 episodes, greedy): avg_return={avg:.1f}")
    print()
    print(format_result([
        ("avg_return", f"{avg:.1f}"),
        ("epochs", str(args.epochs)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()

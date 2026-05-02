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

import torch

from torch_ref.models.mountain_car_cont import (
    Actor,
    MCCState,
    MAX_ACTION,
    MAX_STEPS,
    QNet,
    ReplayBuffer,
    evaluate,
    mcc_step,
    observe,
    polyak_update,
    sac_update,
)
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30000,
                        help="number of env steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--shaping", type=float, default=10.0)
    parser.add_argument("--lr-find", action="store_true",
                        help="Stub for API consistency; SAC's per-step epoch + warmup don't fit lr_find.")
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
    state = MCCState()
    ep_return = 0.0
    ep_len = 0
    for step in range(args.epochs):
        obs = observe(state)
        if step < args.warmup:
            action = rng.uniform(-MAX_ACTION, MAX_ACTION)
        else:
            with torch.no_grad():
                a_t, _ = actor.sample(obs)
                action = float(a_t.item())
        raw_reward, next_state, terminated = mcc_step(state, action)
        ep_return += raw_reward
        ep_len += 1
        truncated = ep_len >= MAX_STEPS
        is_done = terminated or truncated
        buffer_done = terminated
        shaped = raw_reward + args.shaping * abs(next_state.vel)
        buffer.push(obs.tolist(), action, shaped, observe(next_state).tolist(), buffer_done)
        state = next_state
        if is_done:
            history.append(ep_return)
            ep_return = 0.0
            ep_len = 0
            state = MCCState()
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
            print(
                f"  step {step + 1:6d}  eps_seen={len(history):4d}  "
                f"recent_20_return={sum(recent)/len(recent):.1f}"
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

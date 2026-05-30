"""DQN on MountainCar training script.

Aligned with `Example.MountainCar` (Idris).
Usage:
    python -m torch_ref.scripts.mountain_car [--lr 1e-3] [--epochs 1000] [--seed 42]
                                              [--shaping 10.0]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time

import torch

from torch_ref.models.mountain_car import (
    QNetwork,
    ReplayBuffer,
    dqn_episode,
    evaluate,
    make_mountaincar_env,
)
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_elapsed, format_result, mem_suffix, set_device


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
    torch.manual_seed(args.seed)
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
    optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)
    env = make_mountaincar_env(args.seed)
    step_count = [0]
    print()

    def epoch_fn() -> float:
        new_step, ep_return = dqn_episode(
            env,
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

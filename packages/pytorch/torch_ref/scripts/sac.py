"""SAC on Pendulum training script.

Aligned with `Example.Sac` (Idris).
Usage:
    python -m torch_ref.scripts.sac [--lr 3e-4] [--epochs 30000] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from torch_ref.models.sac import evaluate, train_sac
from torch_ref.training.runner import format_result, set_device


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
    parser.add_argument(
        "--lr-find", action="store_true",
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

    torch.manual_seed(args.seed)

    print("=== SAC on Pendulum ===")
    print(
        f"Config: lr={args.lr} steps={args.epochs} gamma={args.gamma}"
        f" alpha={args.alpha} batch={args.batch} buffer={args.buffer}"
        f" warmup={args.warmup} tau={args.tau} seed={args.seed}"
    )
    print()

    t_start = time.monotonic()
    actor, _history = train_sac(
        total_steps=args.epochs, buffer_capacity=args.buffer, batch_size=args.batch,
        lr=args.lr, gamma=args.gamma, alpha=args.alpha,
        warmup_steps=args.warmup, tau=args.tau, seed=args.seed,
    )
    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} steps, {ms_per_ep:.2f}ms/step)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    avg = evaluate(actor, n_episodes=20)
    print(f"Eval (20 episodes, greedy): avg_return={avg:.1f}")
    print()
    print(format_result([
        ("avg_return", f"{avg:.1f}"),
        ("epochs", str(args.epochs)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()

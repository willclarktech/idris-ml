"""REINFORCE on CartPole training script.

Output format matches the Idris Example.Reinforce exactly.

Usage:
    python -m torch_ref.scripts.reinforce [--lr 0.001] [--epochs 2000] [--seed 42]
"""

import argparse
import sys
import time

import torch

from torch_ref.models.reinforce import (
    PolicyNetwork,
    evaluate,
    make_cartpole_env,
    reinforce_epoch,
)
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_elapsed, format_result, mem_suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== REINFORCE on CartPole ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs}"
        f" gamma={args.gamma} batch={args.batch} seed={args.seed}"
    )

    policy = PolicyNetwork(hidden=128)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # setup

    print("Architecture: Linear(4->128)->Tanh->Linear(128->2)")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {n_params}")
    print()

    env = make_cartpole_env(args.seed)

    if args.lr_find:
        def epoch_fn() -> float:
            # `reinforce_epoch` returns (mean episodic return, policy loss).
            # `lr_find` wants a "lower is better" scalar; the Idris example
            # reports `negate avg_return` to runTraining, so we match.
            avg_ret, _ = reinforce_epoch(env, policy, optimizer, args.batch, args.gamma)
            return -avg_ret
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t_start = time.monotonic()
    history: list[float] = []
    for epoch in range(args.epochs):
        avg_return, loss_val = reinforce_epoch(env, policy, optimizer, args.batch, args.gamma)
        history.append(avg_return)
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(
                f"  {format_elapsed(t_start)} {epoch}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={avg_return:.1f}\trecent_100={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} epochs, {ms_per_ep:.0f}ms/epoch)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    print("Eval (100 episodes, greedy):")
    avg_return = evaluate(policy, n_episodes=100)
    print(f"  avg_return={avg_return:.1f}")

    print()
    print(format_result([
        ("avg_return", f"{avg_return:.1f}"),
        ("epochs", str(args.epochs)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()

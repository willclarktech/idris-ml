"""REINFORCE on CartPole training script.

Output format matches the Idris Example.Reinforce exactly.

Usage:
    python -m torch_ref.scripts.reinforce [--lr 0.001] [--epochs 2000] [--seed 42]
"""

import argparse
import time

import torch

from torch_ref.models.reinforce import PolicyNetwork, evaluate, reinforce_epoch
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=10)
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

    print("Training...")
    t0 = time.time()
    for epoch in range(args.epochs):
        loss = reinforce_epoch(policy, optimizer, args.batch, args.gamma)
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"  [{elapsed:07.2f}s] {epoch}\tloss={loss:.6f}")

    elapsed = time.time() - t0
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} epochs, {ms_per_ep:.0f}ms/epoch)")

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

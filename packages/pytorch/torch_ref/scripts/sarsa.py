"""SARSA on CliffWalking training script.

Output format matches the Idris Example.Sarsa exactly.

Usage:
    python -m torch_ref.scripts.sarsa [--alpha 0.5] [--epochs 1000] [--seed 42]
"""

import argparse

from torch_ref.models.sarsa import evaluate, train_sarsa
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== SARSA on CliffWalking ===")
    print(
        f"Config: alpha={args.alpha} gamma={args.gamma}"
        f" epsilon={args.epsilon} epochs={args.epochs} seed={args.seed}"
    )
    print()

    q, history = train_sarsa(
        epochs=args.epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    avg = evaluate(q)
    print()
    print(f"Eval (100 episodes, greedy): avg_return={avg:.1f}")
    print()
    print(
        format_result(
            [
                ("avg_return", f"{avg:.1f}"),
                ("epochs", str(len(history))),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

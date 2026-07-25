"""Q-learning on FrozenLake (slippery 4x4) training script.

Output format matches the Idris Example.FrozenLake exactly.

Usage:
    python -m torch_ref.scripts.frozen_lake [--alpha 0.1] [--epochs 10000] [--seed 42]
"""

import argparse

from torch_ref.models.frozen_lake import evaluate, train_q_learning
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Q-learning on FrozenLake (slippery 4x4) ===")
    print(
        f"Config: alpha={args.alpha} gamma={args.gamma}"
        f" epsilon={args.epsilon} epochs={args.epochs} seed={args.seed}"
    )
    print()

    q, history = train_q_learning(
        epochs=args.epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    avg = evaluate(q)
    print()
    print(f"Eval (100 episodes, greedy): avg_return={avg:.2f}")
    print()
    print(
        format_result(
            [
                ("avg_return", f"{avg:.2f}"),
                ("epochs", str(len(history))),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

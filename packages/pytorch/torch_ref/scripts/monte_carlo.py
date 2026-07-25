"""First-visit Monte Carlo control on Blackjack training script.

Output format matches the Idris Example.MonteCarlo exactly.

Usage:
    python -m torch_ref.scripts.monte_carlo [--epsilon 0.1] [--epochs 50000] [--seed 42]
"""

import argparse

from torch_ref.models.monte_carlo import evaluate, train_mc
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== First-visit MC on Blackjack ===")
    print(f"Config: epsilon={args.epsilon} epochs={args.epochs} seed={args.seed}")
    print()

    q, history = train_mc(epochs=args.epochs, epsilon=args.epsilon, seed=args.seed)
    stats = evaluate(q)
    print()
    print(
        f"Eval (10000 episodes, greedy): win={stats['win']:.3f} draw={stats['draw']:.3f} "
        f"loss={stats['loss']:.3f} avg_reward={stats['avg_reward']:+.3f}"
    )
    print()
    print(
        format_result(
            [
                ("win_rate", f"{stats['win']:.3f}"),
                ("avg_reward", f"{stats['avg_reward']:+.3f}"),
                ("epochs", str(len(history))),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

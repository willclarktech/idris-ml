"""MNIST CNN training script.

Usage:
    python -m torch_ref.scripts.mnist [--lr 0.001] [--epochs 10] [--seed 42]
"""

import argparse
import time

import torch

from torch_ref.models.mnist_cnn import MnistCNN, evaluate, get_mnist_loaders, train_epoch
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== MNIST: Convolutional Neural Network ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs}"
        f" batch_size={args.batch_size} seed={args.seed}"
    )
    print(
        "Architecture: Conv2d(1->16,k=5) -> ReLU -> Pool(2)"
        " -> Conv2d(16->32,k=5) -> ReLU -> Pool(2) -> Linear(512->10)"
    )

    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    model = MnistCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count}")
    print()

    print("Training...")
    t0 = time.time()
    best_loss = float("inf")
    stale = 0
    patience = 500
    epochs_done = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer)
        epochs_done = epoch
        elapsed = time.time() - t0
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            test_loss, accuracy = evaluate(model, test_loader)
            print(
                f"  [{elapsed:07.2f}s] {epoch}\tloss={loss:.6f}"
                f"\ttest_loss={test_loss:.6f}\taccuracy={accuracy * 100:.1f}%"
            )
        # Patience-based early stopping (matches Idris patienceConfig)
        if loss < best_loss - 0.001:
            best_loss = loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    test_loss, accuracy = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"\nFinal: test_loss={test_loss:.6f} accuracy={accuracy * 100:.1f}% ({elapsed:.1f}s)")
    print(format_result([
        ("accuracy", f"{accuracy:.4f}"),
        ("epochs", str(epochs_done)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()

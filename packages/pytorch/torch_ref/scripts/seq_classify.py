"""1D sequence classification training script.

Usage:
    python -m torch_ref.scripts.seq_classify [--lr 0.001] [--epochs 500] [--seed 42]
"""

import argparse
import random
import time

import torch

from torch_ref.models.seq_classify import SeqClassifyCNN, evaluate, train_epoch
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=== SeqClassify: 1D Waveform Classification ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}")
    print(
        "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2)"
        " -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Linear(48->3)"
    )

    model = SeqClassifyCNN().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count}")
    print()

    print("Training...")
    t0 = time.time()
    best_loss = float("inf")
    stale = 0
    patience = 200
    epochs_done = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer)
        epochs_done = epoch
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            accuracy = evaluate(model)
            elapsed = time.time() - t0
            print(
                f"  [{elapsed:07.2f}s] {epoch}"
                f"\tloss={loss:.6f}\taccuracy={accuracy * 100:.1f}%"
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

    accuracy = evaluate(model, 500)
    elapsed = time.time() - t0
    print(f"\nFinal accuracy (500 samples): {accuracy * 100:.1f}% ({elapsed:.1f}s)")
    print(format_result([
        ("accuracy", f"{accuracy:.4f}"),
        ("epochs", str(epochs_done)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()

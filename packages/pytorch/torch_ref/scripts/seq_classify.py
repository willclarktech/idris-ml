"""1D sequence classification training script.

Usage:
    python -m torch_ref.scripts.seq_classify [--lr 0.001] [--epochs 1000] [--seed 42]
"""

import argparse
import random
import sys
import time

import torch

from torch_ref.models.seq_classify import SeqClassifyCNN, evaluate, train_epoch
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_result, get_dtype, set_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
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
    random.seed(args.seed)

    print("=== SeqClassify: 1D Waveform Classification ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}")
    print(
        "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2)"
        " -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Linear(48->3)"
    )

    model = SeqClassifyCNN().to(args.device, dtype=get_dtype())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count}")
    print()

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), lambda: train_epoch(model, optimizer), optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

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

    train_elapsed = time.time() - t0
    ms_per_ep = train_elapsed * 1000.0 / epochs_done if epochs_done > 0 else 0.0
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")
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

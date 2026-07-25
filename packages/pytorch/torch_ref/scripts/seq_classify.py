"""1D sequence classification training script.

Usage:
    python -m torch_ref.scripts.seq_classify [--lr 0.001] [--epochs 1000] [--seed 42]
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from torch_ref.init_manifest import (
    ORACLE_INPUT,
    ORACLE_TARGET,
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.seq_classify import (
    NUM_CLASSES,
    SeqClassifyCNN,
    evaluate,
    generate_batch,
    train_epoch,
)
from torch_ref.replay import write_replay
from torch_ref.training.losses import nll_loss
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_result, get_dtype, set_device

# Idris registry name -> this script's parameter name, model-index prefixed.
# Mirrors the entry in scripts/paired_examples.py, which
# check-step-oracle.py cross-checks.
PAIRED_PARAMS = {
    "conv1d_0.bias": "0.conv1.bias",
    "conv1d_0.weight": "0.conv1.weight",
    "conv1d_1.bias": "0.conv2.bias",
    "conv1d_1.weight": "0.conv2.weight",
    "linear_0.bias": "0.fc.bias",
    "linear_0.weight": "0.fc.weight",
}


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
    # torch's manual_seed stub leaves `seed` unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]
    random.seed(args.seed)

    print("=== SeqClassify: 1D Waveform Classification ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}")
    print(
        "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2)"
        " -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(48->3)"
    )

    model = SeqClassifyCNN().to(args.device, dtype=get_dtype())
    maybe_dump_init(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Oracle run: publish this side's parameters and the exact batch the
    # step consumes (the two sides' waveform generators are not
    # bit-reproducible from one another — quantized vs continuous params —
    # so the batch travels inside the fixture, the supervised.py shape),
    # then take exactly one optimizer step on that batch — train_epoch's op
    # sequence, dropout keep-bits recorded to the replay mask channel — and
    # publish the post-step parameters.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        data, target = generate_batch(32)
        maybe_dump_oracle(
            (model,),
            PAIRED_PARAMS,
            {
                ORACLE_INPUT: data.view(data.size(0), -1),
                ORACLE_TARGET: F.one_hot(target, NUM_CLASSES).to(torch.float64),
            },
        )
        rec: list[str] = []
        model.dropout.recorder = rec
        model.train()
        optimizer.zero_grad()
        loss = nll_loss(model(data), F.one_hot(target, NUM_CLASSES).to(get_dtype()))
        # torch's Tensor.backward / Adam.step stubs are unannotated.
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        model.dropout.recorder = None
        write_replay(os.environ["IDRISML_ORACLE_DUMP"] + ".replay", masks=rec)
        maybe_dump_after_step((model,), PAIRED_PARAMS)

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
    loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer)
        epochs_done = epoch
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            accuracy = evaluate(model)
            elapsed = time.time() - t0
            print(f"  [{elapsed:07.2f}s] {epoch}\tloss={loss:.6f}\taccuracy={accuracy * 100:.1f}%")
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
    print(
        format_result(
            [
                ("accuracy", f"{accuracy:.4f}"),
                ("epochs", str(epochs_done)),
                ("loss", f"{loss:.6f}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

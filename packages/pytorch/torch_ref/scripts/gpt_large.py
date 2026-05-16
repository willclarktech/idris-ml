"""GPT-Large character-level language model training script.

GPU-shaped variant of `torch_ref.scripts.gpt` — same training recipe,
but with the architecture dimensions cranked up so per-step compute is
matmul-dominated (the small `gpt.py` config sits on the wrong side of
mlx's kernel-launch wall, so its GPU-vs-CPU comparison is noise; see
`docs/develop/mlx-survey.md`).

Paired with `packages/idris-ml-examples/src/Example/GptLarge.idr`.

Config: dModel=256, heads=8, headDim=32, blocks=4, seq=128, batch=32
(~1M params vs ~50K for the default `gpt`).

Two corpus paths:
- `--corpus tinyshakespeare` (default for convergence): 1.1 M chars,
  65-char dynamic vocab.
- `--corpus embedded`: 1342-char excerpt, fast wiring test.

Optimizer recipe is aligned with nanoGPT/train_shakespeare_char.py for
the tinyshakespeare path: AdamW β2=0.99 wd=0.1 + cosine LR with warmup.

Usage:
    python -m torch_ref.scripts.gpt_large [--corpus tinyshakespeare|embedded]
                                          [--lr 1e-3] [--epochs 1000] [--seed 42]
"""

import argparse
import math
import random
import sys

import torch

from torch_ref.models.gpt import (
    CORPUS_INDICES,
    VOCAB_SIZE,
    evaluate_bpc,
    generate_gpt_data,
    generate_text,
    load_tinyshakespeare,
    train_gpt_epoch,
    train_val_split,
)
from torch_ref.models.multi_head_transformer import MultiHeadTransformer
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import TrainConfig, format_result, run_training

# Architecture config — GPU-shaped (matmul-dominated). Mirrors the Idris
# Example.GptLarge constants. head_dim = d_model // num_heads = 32.
SEQ_LEN = 128
D_MODEL = 256
NUM_HEADS = 8
NUM_BLOCKS = 4
BATCH_SIZE = 32

# nanoGPT optimizer/schedule defaults (train_shakespeare_char.py).
BETA1 = 0.9
BETA2 = 0.99  # default torch is 0.999; nanoGPT uses 0.99 for char-LM
WEIGHT_DECAY = 0.1  # default 0.01; nanoGPT uses 0.1
GRAD_CLIP = 1.0
MIN_LR_FACTOR = 0.1  # min_lr = base_lr * MIN_LR_FACTOR


def cosine_lr(epoch: int, base_lr: float, max_epochs: int) -> float:
    """Cosine LR schedule with linear warmup. Verbatim nanoGPT formula at
    max_epochs >= 1000; for shorter smoke runs the warmup is capped at
    max_epochs/10 so the LR actually ramps."""
    warmup_epochs = min(100, max_epochs // 10)
    min_lr = base_lr * MIN_LR_FACTOR
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / (warmup_epochs + 1)
    if epoch >= max_epochs:
        return min_lr
    decay_ratio = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["tinyshakespeare", "embedded"], default="embedded")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="0 disables patience; rely on cosine LR for annealing",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Corpus + vocab -----------------------------------------------------
    if args.corpus == "tinyshakespeare":
        text, vocab, all_indices = load_tinyshakespeare()
        train_indices, val_indices = train_val_split(all_indices, val_frac=0.1)
        vocab_size = vocab.size
        corpus_label = (
            f"tinyshakespeare ({len(text)} chars, vocab={vocab_size}, "
            f"train={len(train_indices)}, val={len(val_indices)})"
        )
    else:
        vocab = None
        train_indices = CORPUS_INDICES
        val_indices = CORPUS_INDICES  # smoke path: same set
        vocab_size = VOCAB_SIZE
        corpus_label = f"embedded ({len(CORPUS_INDICES)} chars, vocab={vocab_size})"

    print("=== GPT-Large: Character-Level Language Model ===")
    print(f"Config: corpus={args.corpus} lr={args.lr} epochs={args.epochs} seed={args.seed}")
    print(
        f"Architecture: seqLen={SEQ_LEN} dModel={D_MODEL}"
        f" heads={NUM_HEADS} blocks={NUM_BLOCKS} vocab={vocab_size}"
    )
    print(f"Corpus: {corpus_label}")

    model = MultiHeadTransformer(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")
    print()

    if args.lr_find:

        def lr_find_epoch_fn() -> float:
            data = generate_gpt_data(train_indices, BATCH_SIZE, SEQ_LEN, vocab_size)
            return train_gpt_epoch(model, data, optimizer)

        lr_find(LrFindConfig(num_iters=100), lr_find_epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    epoch_counter = {"i": 0}

    def epoch_fn() -> float:
        ep = epoch_counter["i"]
        lr = cosine_lr(ep, args.lr, args.epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr
        epoch_counter["i"] = ep + 1

        data = generate_gpt_data(train_indices, BATCH_SIZE, SEQ_LEN, vocab_size)
        loss = train_gpt_epoch(model, data, optimizer)
        return loss

    def metrics_fn() -> list[tuple[str, str]]:
        val_bpc = evaluate_bpc(
            model,
            val_indices,
            SEQ_LEN,
            n_samples=20,
            vocab_size=vocab_size,
        )
        return [
            ("val_bpc", f"{val_bpc:.3f}"),
            ("lr", f"{cosine_lr(epoch_counter['i'], args.lr, args.epochs):.5f}"),
        ]

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        min_delta=0.001,
    )

    epochs_done, final_loss = run_training(epoch_fn, config, metrics_fn)

    print()
    val_bpc = evaluate_bpc(
        model,
        val_indices,
        SEQ_LEN,
        n_samples=50,
        vocab_size=vocab_size,
    )
    train_bpc = evaluate_bpc(
        model,
        train_indices,
        SEQ_LEN,
        n_samples=50,
        vocab_size=vocab_size,
    )
    print(f"Final val_bpc: {val_bpc:.3f}  (train_bpc: {train_bpc:.3f})")
    print()

    seed1 = "to be or "
    seed2 = "the "
    print(f"Generation (seed={seed1!r}):")
    sample = generate_text(model, seed1, length=200, temperature=1.0, vocab=vocab)
    print(f"  {sample!r}")
    print()
    print(f"Generation (seed={seed2!r}):")
    sample2 = generate_text(model, seed2, length=200, temperature=1.0, vocab=vocab)
    print(f"  {sample2!r}")

    print()
    metric_key = "val_bpc" if args.corpus == "tinyshakespeare" else "bpc"
    metric_value = val_bpc
    print(
        format_result(
            [
                (metric_key, f"{metric_value:.3f}"),
                ("epochs", str(epochs_done)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

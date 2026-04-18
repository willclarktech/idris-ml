"""GPT character-level language model training script.

Output format matches the Idris Example.Gpt exactly:
banner, per-epoch progress, timing summary, generation sample, RESULT line.

Usage:
    python -m torch_ref.scripts.gpt [--lr 0.001] [--epochs 2000] [--seed 42]
"""

import argparse
import random

import torch

from torch_ref.models.gpt import (
    CORPUS_INDICES,
    VOCAB_SIZE,
    evaluate_bpc,
    generate_gpt_data,
    generate_text,
    train_gpt_epoch,
)
from torch_ref.models.multi_head_transformer import MultiHeadTransformer
from torch_ref.training.runner import TrainConfig, format_result, run_training

# Architecture config matching Idris
SEQ_LEN = 64
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
BATCH_SIZE = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=== GPT: Character-Level Language Model ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} patience={args.patience} seed={args.seed}")
    print(
        f"Architecture: seqLen={SEQ_LEN} dModel={D_MODEL}"
        f" heads={NUM_HEADS} blocks={NUM_BLOCKS} vocab={VOCAB_SIZE}"
    )
    print(f"Corpus: {len(CORPUS_INDICES)} chars")

    model = MultiHeadTransformer(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")
    print()

    def epoch_fn() -> float:
        data = generate_gpt_data(CORPUS_INDICES, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        return train_gpt_epoch(model, data, optimizer)

    def metrics_fn() -> list[tuple[str, str]]:
        bpc = evaluate_bpc(model, CORPUS_INDICES, SEQ_LEN, n_samples=20)
        return [("bpc", f"{bpc:.3f}")]

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        min_delta=0.001,
    )

    epochs_done, final_loss = run_training(epoch_fn, config, metrics_fn)

    # Evaluation
    print()
    bpc = evaluate_bpc(model, CORPUS_INDICES, SEQ_LEN, n_samples=50)
    print(f"Final BPC: {bpc:.3f}")
    print()

    print("Generation (seed='to be or '):")
    sample = generate_text(model, "to be or ", length=100, temperature=0.8)
    print(f"  {sample!r}")

    print()
    print("Generation (seed='the '):")
    sample2 = generate_text(model, "the ", length=100, temperature=0.8)
    print(f"  {sample2!r}")

    print()
    print(
        format_result(
            [
                ("bpc", f"{bpc:.3f}"),
                ("epochs", str(epochs_done)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

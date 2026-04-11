"""Transformer sequence reversal training script.

Output format matches the Idris Example.Transformer exactly:
banner, per-epoch progress, timing summary, evaluation, RESULT line.

Usage:
    python -m torch_ref.scripts.transformer [--lr 0.001] [--epochs 500] [--patience 200] [--seed 42]
"""

import argparse

import torch

from torch_ref.models.multi_head_transformer import (
    MultiHeadTransformer,
    eval_reversal_accuracy,
    generate_reversal_data,
    train_reversal_epoch,
)
from torch_ref.training.runner import TrainConfig, format_result, run_training

# Task config matching Idris
VOCAB_SIZE = 10
INPUT_LEN = 5
SEQ_LEN = 2 * INPUT_LEN + 1
SEP_TOKEN = 8
EOS_TOKEN = 9
D_MODEL = 32
NUM_HEADS = 4
HEAD_DIM = 8
BATCH_SIZE = 16


def token_name(n: int) -> str:
    if n < 8:
        return chr(n + 65)
    if n == SEP_TOKEN:
        return "|"
    if n == EOS_TOKEN:
        return "$"
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== Transformer: Sequence Reversal ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} patience={args.patience} seed={args.seed}")
    print(
        f"Architecture: seqLen={SEQ_LEN} dModel={D_MODEL}"
        f" heads={NUM_HEADS} headDim={HEAD_DIM} vocab={VOCAB_SIZE}"
    )

    model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model: Transformer<{SEQ_LEN}x{D_MODEL} h={NUM_HEADS} v={VOCAB_SIZE}>")
    print()

    # Training with fresh data each epoch
    def epoch_fn() -> float:
        data = generate_reversal_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        return train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

    def metrics_fn() -> list[tuple[str, str]]:
        eval_data = generate_reversal_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        _, rev_acc = eval_reversal_accuracy(model, eval_data, INPUT_LEN)
        rev_correct = int(rev_acc * BATCH_SIZE * (SEQ_LEN - INPUT_LEN))
        rev_total = BATCH_SIZE * (SEQ_LEN - INPUT_LEN)
        return [("rev_acc", f"{rev_correct}/{rev_total}")]

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        min_delta=0.001,
    )

    epochs_done, final_loss = run_training(epoch_fn, config, metrics_fn)

    # Evaluation on a fresh example
    print()
    print("Evaluation:")
    eval_data = generate_reversal_data(1, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
    inp_onehot, target_indices = eval_data[0]
    with torch.no_grad():
        logits = model(inp_onehot)
        preds = logits.argmax(dim=-1)

    input_decoded = inp_onehot.argmax(dim=-1).tolist()
    target_decoded = target_indices.tolist()
    predicted = preds.tolist()

    input_tokens = input_decoded[:INPUT_LEN]
    rev_target = target_decoded[INPUT_LEN:]
    rev_predicted = predicted[INPUT_LEN:]
    print(f"  Input:      {''.join(token_name(t) for t in input_tokens)}")
    print(f"  Target:     {''.join(token_name(t) for t in rev_target)}")
    print(f"  Predicted:  {''.join(token_name(t) for t in rev_predicted)}")

    rev_correct = sum(
        1 for p, t in zip(predicted[INPUT_LEN:], target_decoded[INPUT_LEN:], strict=True) if p == t
    )
    rev_total = SEQ_LEN - INPUT_LEN

    print(f"  Rev acc:    {rev_correct}/{rev_total}")

    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("rev_acc", f"{rev_correct}/{rev_total}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

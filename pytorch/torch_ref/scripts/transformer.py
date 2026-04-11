"""Transformer sequence sorting training script.

Output format matches the Idris Example.Transformer exactly:
banner, per-epoch progress, timing summary, evaluation, RESULT line.

Usage:
    python -m torch_ref.scripts.transformer [--lr 0.001] [--epochs 1000] [--blocks 2] [--seed 42]
"""

import argparse

import torch

from torch_ref.models.multi_head_transformer import (
    MultiHeadTransformer,
    eval_reversal_accuracy,
    generate_sorting_data,
    train_reversal_epoch,
)
from torch_ref.training.runner import TrainConfig, format_result, run_training

# Task config matching Idris
VOCAB_SIZE = 8  # digits 0-5 + SEP + EOS
INPUT_LEN = 5
SEQ_LEN = 2 * INPUT_LEN + 1  # 11
SEP_TOKEN = 6
EOS_TOKEN = 7
D_MODEL = 32
NUM_HEADS = 4
HEAD_DIM = 8
BATCH_SIZE = 16


def token_name(n: int) -> str:
    if n < 6:
        return str(n)
    if n == SEP_TOKEN:
        return "|"
    if n == EOS_TOKEN:
        return "$"
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== Transformer: Sequence Sorting ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} patience={args.patience} seed={args.seed}")
    print(
        f"Architecture: seqLen={SEQ_LEN} dModel={D_MODEL}"
        f" heads={NUM_HEADS} headDim={HEAD_DIM}"
        f" blocks={args.blocks} vocab={VOCAB_SIZE}"
    )

    model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS, num_blocks=args.blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        f"Model: Transformer<{SEQ_LEN}x{D_MODEL} h={NUM_HEADS} blocks={args.blocks} v={VOCAB_SIZE}>"
    )
    print()

    def epoch_fn() -> float:
        data = generate_sorting_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        return train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

    def metrics_fn() -> list[tuple[str, str]]:
        eval_data = generate_sorting_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        _, acc = eval_reversal_accuracy(model, eval_data, INPUT_LEN)
        correct = int(acc * BATCH_SIZE * (SEQ_LEN - INPUT_LEN))
        total = BATCH_SIZE * (SEQ_LEN - INPUT_LEN)
        return [("sort_acc", f"{correct}/{total}")]

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        min_delta=0.001,
    )

    epochs_done, final_loss = run_training(epoch_fn, config, metrics_fn)

    # Evaluation
    print()
    print("Evaluation:")
    eval_data = generate_sorting_data(1, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
    inp_onehot, target_indices = eval_data[0]
    with torch.no_grad():
        logits = model(inp_onehot)
        preds = logits.argmax(dim=-1)

    input_decoded = inp_onehot.argmax(dim=-1).tolist()
    target_decoded = target_indices.tolist()
    predicted = preds.tolist()

    input_tokens = input_decoded[:INPUT_LEN]
    sort_target = target_decoded[INPUT_LEN:]
    sort_predicted = predicted[INPUT_LEN:]
    print(f"  Input:      {''.join(token_name(t) for t in input_tokens)}")
    print(f"  Target:     {''.join(token_name(t) for t in sort_target)}")
    print(f"  Predicted:  {''.join(token_name(t) for t in sort_predicted)}")

    sort_correct = sum(
        1 for p, t in zip(predicted[INPUT_LEN:], target_decoded[INPUT_LEN:], strict=True) if p == t
    )
    sort_total = SEQ_LEN - INPUT_LEN

    print(f"  Sort acc:   {sort_correct}/{sort_total}")

    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("sort_acc", f"{sort_correct}/{sort_total}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

"""LSTM pattern prediction training script.

Output format matches Idris Example.Lstm.
Same pattern task as RNN but with LSTM cell + patience early stopping.
"""

import argparse
import os
import sys
from typing import cast

import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.rnn import LinearLSTMCell, generate_rnn_dataset, train_lstm_epoch
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import TrainConfig, format_result, run_training, set_device

# Idris registry name -> this script's parameter name. Mirrors the entry in
# scripts/paired_examples.py, which check-step-oracle.py cross-checks.
PAIRED_PARAMS = {
    "linear_0.bias": "0.output_proj.bias",
    "linear_0.weight": "0.output_proj.weight",
    "lstm_0.bias_hh": "0.lstm.bias_hh",
    "lstm_0.bias_ih": "0.lstm.bias_ih",
    "lstm_0.c0": "0.c0",
    "lstm_0.h0": "0.h0",
    "lstm_0.weight_hh": "0.lstm.weight_hh",
    "lstm_0.weight_ih": "0.lstm.weight_ih",
}


def show_seq(tensors: list[torch.Tensor]) -> str:
    return "[" + ",".join(str(int(t.item() > 0)) for t in tensors) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=500)
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

    print("=== LSTM Pattern Prediction ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} patience={args.patience} seed={args.seed}")

    model = LinearLSTMCell(1, 4, 1).to(args.device)
    maybe_dump_init(model)
    data = generate_rnn_dataset(8)  # built under active device set above
    print("Architecture: LstmLayer ~> OutputLayer (LinearLayer)")
    print()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def epoch_fn() -> float:
        return train_lstm_epoch(model, data, optimizer)

    # Oracle run: publish this side's parameters, then take exactly one epoch
    # (which is one optimizer step — `generate` streams carry no epoch length,
    # so the Idris loop also steps once per epoch) and publish the result.
    # `generate_rnn_dataset` produces the same fixed sequences Idris'
    # `patternSeqs` does, so only the weights travel.
    maybe_dump_oracle((model,), PAIRED_PARAMS)
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        epoch_fn()
        maybe_dump_after_step((model,), PAIRED_PARAMS)

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        device=args.device,
    )
    epochs_done, _final_loss = run_training(epoch_fn, config)

    # Evaluation
    print()
    print("Eval:")

    eval_loss = 0.0
    predictions: list[list[torch.Tensor]] = []
    targets: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for xs, ys in data:
            model.reset_state()
            preds: list[torch.Tensor] = []
            for x in xs:
                pred = model(x)
                preds.append(pred)
            predictions.append(preds)
            targets.append(ys)
            from torch_ref.training.losses import bce_with_logits

            seq_loss = sum(bce_with_logits(p, y) for p, y in zip(preds, ys, strict=True))
            # Sequences are non-empty, so the sum is a Tensor (never the int 0 start).
            eval_loss += cast("torch.Tensor", seq_loss).item() / len(xs)
        eval_loss /= len(data)

    print(f"  Loss: {eval_loss}")
    print("  Seq  Target     Predicted")
    for i, (tgt, pred) in enumerate(zip(targets, predictions, strict=True)):
        ts = show_seq(tgt)
        ps = show_seq(pred)
        ok = " ok" if ts == ps else ""
        print(f"  {i + 1}.   {ts}  ->  {ps}{ok}")

    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("loss", str(eval_loss)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

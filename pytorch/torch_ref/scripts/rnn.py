"""RNN pattern prediction training script.

Output format matches Idris Example.Rnn.
Pattern: [0,1,0,0,1,0,...] — predict next element.
"""

import argparse

import torch

from torch_ref.models.rnn import LinearRNNCell, generate_rnn_dataset, train_rnn_epoch
from torch_ref.training.runner import TrainConfig, format_result, run_training


def show_seq(tensors: list[torch.Tensor]) -> str:
    return "[" + ",".join(str(int(t.item() > 0)) for t in tensors) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== RNN Pattern Prediction ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}")

    model = LinearRNNCell(1, 1)
    data = generate_rnn_dataset(8)
    print("Architecture: OutputLayer (RnnLayer)")
    print()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def epoch_fn() -> float:
        return train_rnn_epoch(model, data, optimizer)

    config = TrainConfig(total_epochs=args.epochs, log_every=100)
    epochs_done, final_loss = run_training(epoch_fn, config)

    # Evaluation
    print()
    print("Eval:")

    eval_loss = 0.0
    predictions: list[list[torch.Tensor]] = []
    targets: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for xs, ys in data:
            model.reset_state()
            preds = []
            for x in xs:
                pred = model(x)
                preds.append(pred)
            predictions.append(preds)
            targets.append(ys)
            from torch_ref.training.losses import bce_with_logits

            seq_loss = sum(bce_with_logits(p, y) for p, y in zip(preds, ys, strict=True))
            eval_loss += seq_loss.item() / len(xs)  # type: ignore[union-attr]
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

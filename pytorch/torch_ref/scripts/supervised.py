"""Supervised classification training script.

Output format matches Idris Example.Supervised.
"""

import argparse

import torch

from torch_ref.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch
from torch_ref.training.runner import TrainConfig, format_result, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== Supervised Classification ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}")

    model = SupervisedModel()
    data = SUPERVISED_DATA
    print("Model: Linear<2:3> -> softmax")
    print()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def epoch_fn() -> float:
        return train_supervised_epoch(model, data, optimizer)

    config = TrainConfig(total_epochs=args.epochs, log_every=100)
    epochs_done, final_loss = run_training(epoch_fn, config)

    # Evaluation
    print()
    print("Eval:")
    with torch.no_grad():
        from torch_ref.training.losses import cross_entropy

        losses = torch.stack([cross_entropy(model(x), y) for x, y in data])
        eval_loss = losses.mean().item()
    print(f"  Loss: {eval_loss}")

    with torch.no_grad():
        for x, y in data:
            pred = model(x)
            pred_class = pred.argmax().item()
            target_class = y.argmax().item()
            ok = "ok" if pred_class == target_class else "WRONG"
            print(f"  {x.tolist()} -> class {pred_class} {ok}")

    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("loss", f"{eval_loss}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

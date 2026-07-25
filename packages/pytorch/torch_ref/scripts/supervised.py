"""Supervised classification training script.

Output format matches Idris Example.Supervised.
"""

import argparse
import sys
from typing import cast

import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_batch,
    maybe_dump_init,
    maybe_load_oracle,
)
from torch_ref.models.supervised import (
    SupervisedModel,
    _make_supervised_data,  # pyright: ignore[reportPrivateUsage]  # shared with the paired script
    train_supervised_epoch,
)
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    TrainConfig,
    format_result,
    get_dtype,
    run_training,
    set_device,
)

# Idris registry name -> this script's parameter name. Mirrors the entry in
# scripts/paired_examples.py, which is what the alignment gates verify.
PAIRED_PARAMS = {
    "linear_0.bias": "0.linear.bias",
    "linear_0.weight": "0.linear.weight",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.03)
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

    print("=== Supervised Classification ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}")

    model = SupervisedModel().to(args.device, dtype=get_dtype())
    maybe_dump_init(model)
    data = _make_supervised_data()  # rebuilt under active device/dtype
    # The whole 5-sample dataset as one batch, matching Idris `batched {b=5}`.
    maybe_dump_batch(
        torch.stack([x for x, _ in data]),
        torch.stack([y for _, y in data]),
    )
    print("Model: Linear<2:3> -> softmax")
    print()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Oracle run: take Idris' init weights and its batch, do exactly one step,
    # dump the result. Both sides then started from identical numbers on
    # identical inputs, so any difference afterwards is forward, backward or
    # optimizer semantics — the class no shape or moment check can see.
    oracle = maybe_load_oracle(model, PAIRED_PARAMS)
    if oracle is not None:
        ox, oy = oracle
        data = [(ox[i], oy[i]) for i in range(ox.shape[0])]

    def epoch_fn() -> float:
        return train_supervised_epoch(model, data, optimizer)

    if oracle is not None:
        epoch_fn()
        maybe_dump_after_step(model)

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    config = TrainConfig(total_epochs=args.epochs, log_every=100, device=args.device)
    epochs_done, _final_loss = run_training(epoch_fn, config)

    # Evaluation
    print()
    print("Eval:")
    with torch.no_grad():
        import torch.nn.functional as F

        from torch_ref.training.losses import nll_loss

        losses = torch.stack([nll_loss(F.log_softmax(model(x), dim=-1), y) for x, y in data])
        eval_loss = losses.mean().item()
    print(f"  Loss: {eval_loss}")

    correct = 0
    with torch.no_grad():
        for x, y in data:
            pred = model(x)
            pred_class = pred.argmax().item()
            target_class = y.argmax().item()
            correct += int(pred_class == target_class)
            ok = "ok" if pred_class == target_class else "WRONG"
            # tolist() returns list[Unknown] in torch's stubs.
            x_vals = cast("list[float]", x.tolist())  # pyright: ignore[reportUnknownMemberType]
            print(f"  {x_vals} -> class {pred_class} {ok}")

    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("loss", f"{eval_loss}"),
                ("seed", str(args.seed)),
                ("correct", f"{correct}/{len(data)}"),
            ]
        )
    )


if __name__ == "__main__":
    main()

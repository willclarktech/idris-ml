"""NTM copy task training script.

Output format matches Idris Example.NtmCopy.
"""

import argparse
import os
import random
import sys
from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from torch_ref.data.copy_task import generate_copy_batch
from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.metrics import bit_and_sequence_accuracy
from torch_ref.models.ntm import NtmConfig, NtmModel
from torch_ref.replay import write_replay
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    TrainConfig,
    format_result,
    get_device,
    get_dtype,
    run_training,
    set_device,
)

# Architecture constants matching Idris
W = 8
N, M, H = 128, 20, 100
INPUT_W = W + 1
OUTPUT_W = W


# Idris registry name -> this script's parameter name, model-index prefixed.
# Mirrors the entry in scripts/paired_examples.py, which check-step-oracle.py
# cross-checks.
PAIRED_PARAMS = {
    "ntm_0.controller.bias_hh": "0.ntm.controller.lstm.bias_hh",
    "ntm_0.controller.bias_ih": "0.ntm.controller.lstm.bias_ih",
    "ntm_0.controller.c0": "0.ntm.controller.c0",
    "ntm_0.controller.h0": "0.ntm.controller.h0",
    "ntm_0.controller.weight_hh": "0.ntm.controller.lstm.weight_hh",
    "ntm_0.controller.weight_ih": "0.ntm.controller.lstm.weight_ih",
    "ntm_0.memory_init_0": "0.ntm.memory_init",
    "ntm_0.output_fc.bias": "0.ntm.output_fc.bias",
    "ntm_0.read_init_0": "0.ntm.read_init",
    "ntm_0.output_fc.weight": "0.ntm.output_fc.weight",
    "ntm_0.read_fc.bias": "0.ntm.read_fc.bias",
    "ntm_0.read_fc.weight": "0.ntm.read_fc.weight",
    "ntm_0.write_fc.bias": "0.ntm.write_fc.bias",
    "ntm_0.write_fc.weight": "0.ntm.write_fc.weight",
}


def _train_ntm_epoch(
    model: NtmModel,
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    clip_value: float,
) -> float:
    optimizer.zero_grad()
    device = get_device()
    total_loss = torch.tensor(0.0, device=device)
    for input_seq, target_seq in batch:
        model.reset_state()
        seq_len = target_seq.shape[0]
        input_width = input_seq.shape[1]
        for t in range(input_seq.shape[0]):
            model(input_seq[t])
        zero_input = torch.zeros(input_width, device=device)
        outputs: list[torch.Tensor] = []
        for _ in range(seq_len):
            out = model(zero_input)
            outputs.append(out)
        pred = torch.stack(outputs)
        loss = F.binary_cross_entropy_with_logits(pred, target_seq)
        total_loss = total_loss + loss
    avg_loss = total_loss / len(batch)
    # torch's Tensor.backward stub leaves its params unannotated.
    avg_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    clip_grad_value_(model.parameters(), clip_value)
    optimizer.step()
    return avg_loss.item()


def _accuracies(
    model: NtmModel, batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[float, float]:
    """(per-bit, per-sequence) accuracy over a batch."""
    device = get_device()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for input_seq, target_seq in batch:
            model.reset_state()
            for t in range(input_seq.shape[0]):
                model(input_seq[t])
            zero_input = torch.zeros(input_seq.shape[1], device=device)
            outputs: list[torch.Tensor] = []
            for _ in range(target_seq.shape[0]):
                out = model(zero_input)
                outputs.append(out)
            preds.append((torch.sigmoid(torch.stack(outputs)) >= 0.5).float())
            targets.append(target_seq)
    return bit_and_sequence_accuracy(preds, targets)


def show_binary_vec(t: torch.Tensor) -> str:
    return "".join(str(int(x.item())) for x in t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--es-threshold", type=float, default=0.01)
    parser.add_argument("--es-window", type=int, default=1000)
    parser.add_argument("--es-patience", type=int, default=3)
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
    # The ntm/dnc module trees build bare (default-dtype) tensors throughout;
    # pin the process default so they construct in the training dtype.
    torch.set_default_dtype(get_dtype())  # pyright: ignore[reportUnknownMemberType]
    random.seed(args.seed)
    # torch's manual_seed stub leaves `seed` unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]

    print("=== NTM Copy ===")
    print(
        f"Config: lr={args.lr} clip={args.clip} epochs={args.epochs}"
        f" seed={args.seed} batch={args.batch} seqLen={args.min_len}-{args.max_len}"
    )
    print(f"Architecture: N={N} M={M} H={H}")

    cfg = NtmConfig(input_width=INPUT_W, output_width=OUTPUT_W, n=N, m=M, controller_size=H)
    model = NtmModel(cfg).to(args.device)
    maybe_dump_init(model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, momentum=0.9)
    print(f"Model: NTM<N={N} M={M} H={H}>")
    print()

    # Oracle run: publish the parameters and the batch's draws — each
    # sequence's length, then its bits row-major, recovered from the target
    # (the order the Idris side draws them) — take exactly one update and
    # publish the result. Idris replays the draws through --replay and
    # rebuilds the identical batch; sequence construction, the encode/decode
    # recurrence, BCE, the value clip and RMSprop are all under test.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        batch = generate_copy_batch(args.batch, args.min_len, args.max_len, seq_width=W)
        choices: list[int] = []
        for _inp, tgt in batch:
            choices.append(int(tgt.shape[0]))
            choices.extend(int(b) for b in cast("list[float]", tgt.reshape(-1).tolist()))  # pyright: ignore[reportUnknownMemberType]
        maybe_dump_oracle((model,), PAIRED_PARAMS)
        write_replay(os.environ["IDRISML_ORACLE_DUMP"] + ".replay", choices=choices)
        _train_ntm_epoch(model, batch, optimizer, args.clip)
        maybe_dump_after_step((model,), PAIRED_PARAMS)

    def epoch_fn() -> float:
        batch = generate_copy_batch(args.batch, args.min_len, args.max_len, seq_width=W)
        return _train_ntm_epoch(model, batch, optimizer, args.clip)

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    def metrics_fn() -> list[tuple[str, str]]:
        eval_batch = generate_copy_batch(10, 1, 20, seq_width=W)
        acc, _ = _accuracies(model, eval_batch)
        return [
            ("acc", f"{acc * 100:.1f}%"),
        ]

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        windowed_threshold=args.es_threshold,
        windowed_window=args.es_window,
        windowed_patience=args.es_patience,
        windowed_percentile=0.10,
        device=args.device,
    )
    epochs_done, _ = run_training(epoch_fn, config, metrics_fn)

    # Evaluation
    print()
    print("Eval:")
    sample_batch = generate_copy_batch(2, 3, 5, seq_width=W)
    with torch.no_grad():
        for input_seq, target_seq in sample_batch:
            model.reset_state()
            for t in range(input_seq.shape[0]):
                model(input_seq[t])
            zero_input = torch.zeros(INPUT_W, device=get_device())
            outputs: list[torch.Tensor] = []
            for _ in range(target_seq.shape[0]):
                out = model(zero_input)
                outputs.append(out)
            pred = torch.stack(outputs)
            pred_bits = (torch.sigmoid(pred) >= 0.5).float()
            n = target_seq.shape[0]
            inp_str = " ".join(show_binary_vec(input_seq[t, :W]) for t in range(n))
            tgt_str = " ".join(show_binary_vec(target_seq[t]) for t in range(n))
            out_str = " ".join(show_binary_vec(pred_bits[t]) for t in range(n))
            print(f"  Input:  {inp_str}")
            print(f"  Target: {tgt_str}")
            print(f"  Output: {out_str}")
            print()

    test_size = 100
    short_batch = generate_copy_batch(test_size, 1, 5, seq_width=W)
    full_batch = generate_copy_batch(test_size, 1, 20, seq_width=W)
    short_acc, short_seq = _accuracies(model, short_batch)
    full_acc, full_seq = _accuracies(model, full_batch)

    print(f"  Short (len 1-5):  {short_acc * 100:.1f}% bit, {short_seq * 100:.1f}% seq")
    print(f"  Full  (len 1-20): {full_acc * 100:.1f}% bit, {full_seq * 100:.1f}% seq")
    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("acc_short", f"{short_acc}"),
                ("acc_full", f"{full_acc}"),
                ("seq_acc_short", f"{short_seq}"),
                ("seq_acc_full", f"{full_seq}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

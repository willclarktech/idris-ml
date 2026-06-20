"""DNC copy task training script.

Output format matches Idris example conventions.
"""

import argparse
import random
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from torch_ref.data.copy_task import generate_copy_batch
from torch_ref.metrics import bit_and_sequence_accuracy
from torch_ref.models.dnc import DncConfig, DncModel
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    TrainConfig,
    format_result,
    get_device,
    run_training,
    set_device,
)

# Architecture constants matching Idris
W = 8
N, M, H = 32, 20, 100
R = 1  # read heads
INPUT_W = W + 1
OUTPUT_W = W


def _train_dnc_epoch(
    model: DncModel,
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
    model: DncModel, batch: list[tuple[torch.Tensor, torch.Tensor]]
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
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--es-threshold", type=float, default=0.01)
    parser.add_argument("--es-window", type=int, default=1000)
    parser.add_argument("--es-patience", type=int, default=3)
    parser.add_argument("--num-reads", type=int, default=R)
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
    random.seed(args.seed)
    # torch's manual_seed stub leaves `seed` unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]

    print("=== DNC Copy ===")
    print(
        f"Config: lr={args.lr} clip={args.clip} epochs={args.epochs}"
        f" seed={args.seed} batch={args.batch} seqLen={args.min_len}-{args.max_len}"
    )
    print(f"Architecture: N={N} M={M} H={H} R={args.num_reads}")

    cfg = DncConfig(
        input_width=INPUT_W,
        output_width=OUTPUT_W,
        n=N,
        m=M,
        num_reads=args.num_reads,
        controller_size=H,
    )
    model = DncModel(cfg).to(args.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, momentum=0.9)
    print(f"Model: DNC<N={N} M={M} H={H} R={args.num_reads}>")
    print()

    def epoch_fn() -> float:
        batch = generate_copy_batch(args.batch, args.min_len, args.max_len, seq_width=W)
        return _train_dnc_epoch(model, batch, optimizer, args.clip)

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

    test_size = 20
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

"""NTM associative recall training script.

Output format matches Idris Example.NtmAssociativeRecall.
"""

import argparse
import random
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from torch_ref.data.recall_task import generate_recall_batch
from torch_ref.models.ntm import NtmConfig, NtmModel
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    TrainConfig,
    format_result,
    get_device,
    run_training,
    set_device,
)

# Architecture constants matching Idris
W = 6
SEQ_LEN = 3
N, M, H = 128, 20, 100
INPUT_W = W + 2
OUTPUT_W = W


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
        outputs = []
        for _ in range(seq_len):
            out = model(zero_input)
            outputs.append(out)
        pred = torch.stack(outputs)
        loss = F.binary_cross_entropy_with_logits(pred, target_seq)
        total_loss = total_loss + loss
    avg_loss = total_loss / len(batch)
    avg_loss.backward()
    clip_grad_value_(model.parameters(), clip_value)
    optimizer.step()
    return avg_loss.item()


def _bit_accuracy(model: NtmModel, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    correct = 0
    total = 0
    device = get_device()
    with torch.no_grad():
        for input_seq, target_seq in batch:
            model.reset_state()
            for t in range(input_seq.shape[0]):
                model(input_seq[t])
            zero_input = torch.zeros(input_seq.shape[1], device=device)
            outputs = []
            for _ in range(target_seq.shape[0]):
                out = model(zero_input)
                outputs.append(out)
            pred = torch.stack(outputs)
            pred_bits = (torch.sigmoid(pred) >= 0.5).float()
            correct += (pred_bits == target_seq).sum().item()
            total += target_seq.numel()
    return correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--min-items", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=6)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=== NTM Associative Recall ===")
    print(
        f"Config: lr={args.lr} clip={args.clip} epochs={args.epochs}"
        f" seed={args.seed} batch={args.batch}"
        f" items={args.min_items}-{args.max_items} seqLen={SEQ_LEN}"
    )
    print(f"Architecture: N={N} M={M} H={H}")

    cfg = NtmConfig(input_width=INPUT_W, output_width=OUTPUT_W, n=N, m=M, controller_size=H)
    model = NtmModel(cfg).to(args.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, momentum=0.9)
    print(f"Model: NTM<N={N} M={M} H={H}>")
    print()

    def epoch_fn() -> float:
        batch = generate_recall_batch(args.batch, args.min_items, args.max_items, SEQ_LEN, W)
        return _train_ntm_epoch(model, batch, optimizer, args.clip)

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    def metrics_fn() -> list[tuple[str, str]]:
        eval_batch = generate_recall_batch(10, args.min_items, args.max_items, SEQ_LEN, W)
        acc = _bit_accuracy(model, eval_batch)
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
    test_size = 100
    k2_batch = generate_recall_batch(test_size, 2, 2, SEQ_LEN, W)
    k4_batch = generate_recall_batch(test_size, 4, 4, SEQ_LEN, W)
    k6_batch = generate_recall_batch(test_size, 6, 6, SEQ_LEN, W)
    k2_acc = _bit_accuracy(model, k2_batch)
    k4_acc = _bit_accuracy(model, k4_batch)
    k6_acc = _bit_accuracy(model, k6_batch)

    print()
    print("Eval:")
    print(f"  K=2 items: {k2_acc * 100:.1f}% bit accuracy")
    print(f"  K=4 items: {k4_acc * 100:.1f}% bit accuracy")
    print(f"  K=6 items: {k6_acc * 100:.1f}% bit accuracy")
    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("acc_k2", f"{k2_acc}"),
                ("acc_k4", f"{k4_acc}"),
                ("acc_k6", f"{k6_acc}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()

"""NTM convergence verification matching reference implementations.

Copy task: loudinthecloud/pytorch-ntm (50K iterations, RMSprop, BCELoss)
Recall task: vlgiitr/Graves 2014 (100K iterations, RMSprop, BCELoss)

Usage:
    uv run python -m bench.scripts.convergence [--task {copy,recall,both}] [--seed N]
"""

import argparse
import random

import torch

from bench.data.copy_task import generate_copy_sequence
from bench.data.recall_task import generate_recall_sequence
from bench.models.ntm_copy import NtmCopyConfig, NtmCopyModel, train_ntm_copy_step
from bench.models.ntm_recall import NtmRecallConfig, NtmRecallModel, train_ntm_recall_step

# ---------------------------------------------------------------------------
# Copy task
# ---------------------------------------------------------------------------


def run_copy(args: argparse.Namespace) -> None:
    """Train NTM on copy task (loudinthecloud reference)."""
    print("=" * 60)
    print("NTM Copy Task Convergence")
    print("=" * 60)

    cfg = NtmCopyConfig(iterations=getattr(args, "copy_iters", 50000))
    model = NtmCopyModel(cfg)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

    print(f"  seq_width={cfg.seq_width}  seq_range=[{cfg.seq_min},{cfg.seq_max}]")
    print(f"  N={cfg.n}  M={cfg.m}  controller={cfg.controller_size}")
    print(f"  lr={cfg.lr}  clip={cfg.clip_value}  iterations={cfg.iterations}")

    # Training loop: 1 sequence per iteration
    losses: list[float] = []
    for i in range(1, cfg.iterations + 1):
        input_seq, target_seq = generate_copy_sequence(
            seq_len=random.randint(cfg.seq_min, cfg.seq_max),
            seq_width=cfg.seq_width,
        )
        loss = train_ntm_copy_step(model, input_seq, target_seq, optimizer)
        losses.append(loss)

        if i % 500 == 0:
            avg_loss = sum(losses[-500:]) / len(losses[-500:])
            print(f"  iter {i:6d}: loss={avg_loss:.6f}")

    # Evaluate
    print("\n--- Evaluation ---")
    model.eval()
    for test_len in [5, 10, 20]:
        correct_bits = 0
        total_bits = 0
        for _ in range(10):
            input_seq, target_seq = generate_copy_sequence(test_len, cfg.seq_width)
            with torch.no_grad():
                model.reset_state()
                # Input phase
                for t in range(input_seq.shape[0]):
                    model(input_seq[t])
                # Output phase
                outputs = []
                for _ in range(test_len):
                    out = model(torch.zeros(input_seq.shape[1]))
                    outputs.append(out)
                pred = torch.stack(outputs)
                pred_bits = (pred > 0.5).float()
                correct_bits += (pred_bits == target_seq).sum().item()
                total_bits += target_seq.numel()

        accuracy = correct_bits / total_bits if total_bits > 0 else 0
        print(f"  Length {test_len:2d}: bit accuracy = {accuracy:.1%}")


# ---------------------------------------------------------------------------
# Recall task
# ---------------------------------------------------------------------------


def run_recall(args: argparse.Namespace) -> None:
    """Train NTM on associative recall task (Graves 2014 / vlgiitr)."""
    print("=" * 60)
    print("NTM Associative Recall Convergence")
    print("=" * 60)

    cfg = NtmRecallConfig(iterations=getattr(args, "recall_iters", 100000))
    model = NtmRecallModel(cfg)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

    print(f"  seq_width={cfg.seq_width}  seq_len={cfg.seq_len}")
    print(f"  items=[{cfg.min_items},{cfg.max_items}]")
    print(f"  N={cfg.n}  M={cfg.m}  controller={cfg.controller_size}")
    print(f"  lr={cfg.lr}  clip={cfg.clip_value}  iterations={cfg.iterations}")

    # Training loop: 1 sequence per iteration
    losses: list[float] = []
    for i in range(1, cfg.iterations + 1):
        num_items = random.randint(cfg.min_items, cfg.max_items)
        input_seq, target_seq = generate_recall_sequence(
            num_items=num_items,
            seq_len=cfg.seq_len,
            seq_width=cfg.seq_width,
        )
        loss = train_ntm_recall_step(model, input_seq, target_seq, optimizer)
        losses.append(loss)

        if i % 500 == 0:
            avg_loss = sum(losses[-500:]) / len(losses[-500:])
            print(f"  iter {i:6d}: loss={avg_loss:.6f}")

    # Evaluate
    print("\n--- Evaluation ---")
    model.eval()
    for test_items in [2, 3, 4, 5, 6]:
        correct_bits = 0
        total_bits = 0
        for _ in range(10):
            input_seq, target_seq = generate_recall_sequence(test_items, cfg.seq_len, cfg.seq_width)
            with torch.no_grad():
                model.reset_state()
                seq_len = target_seq.shape[0]

                # Feed entire input
                for t in range(input_seq.shape[0]):
                    model(input_seq[t])

                # Output phase: feed zeros
                zero_input = torch.zeros(input_seq.shape[1])
                outputs = []
                for _ in range(seq_len):
                    out = model(zero_input)
                    outputs.append(out)

                pred = torch.stack(outputs)
                pred_bits = (pred > 0.5).float()
                correct_bits += (pred_bits == target_seq).sum().item()
                total_bits += target_seq.numel()

        accuracy = correct_bits / total_bits if total_bits > 0 else 0
        print(f"  {test_items} items: bit accuracy = {accuracy:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="NTM convergence verification")
    parser.add_argument(
        "--task",
        choices=["copy", "recall", "both"],
        default="both",
        help="Which task to run (default: both)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--copy-iters", type=int, default=50000, help="Copy task iterations")
    parser.add_argument("--recall-iters", type=int, default=100000, help="Recall task iterations")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.task in ("copy", "both"):
        run_copy(args)
        print()

    if args.task in ("recall", "both"):
        run_recall(args)


if __name__ == "__main__":
    main()

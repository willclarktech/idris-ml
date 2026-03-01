"""NTM convergence verification matching reference implementations.

Copy task: loudinthecloud/pytorch-ntm (50K iterations, RMSprop, BCELoss)
Recall task: vlgiitr/Graves 2014 (100K iterations, RMSprop, BCELoss)

Usage:
    uv run python -m bench.scripts.convergence [--task {copy,recall,both}] [--seed N]

    # Recall with Adam optimizer:
    uv run python -m bench.scripts.convergence --task recall \
        --recall-optimizer adam --recall-lr 1e-3

    # Recall with small memory and RNN controller:
    uv run python -m bench.scripts.convergence --task recall \
        --recall-controller rnn --recall-n 16
"""

import argparse
import random
from typing import Literal

import torch

from bench.data.copy_task import generate_copy_sequence
from bench.data.recall_task import generate_recall_sequence
from bench.diagnostics.ntm_diagnostics import (
    compute_summary,
    instrumented_forward_recall,
    print_summary,
)
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

    clip_mode: Literal["value", "norm"] = getattr(args, "copy_clip", "value")
    optimizer_name: str = getattr(args, "copy_optimizer", "rmsprop")
    lr: float = getattr(args, "copy_lr", 1e-4)

    cfg = NtmCopyConfig(
        iterations=getattr(args, "copy_iters", 50000),
        n=getattr(args, "copy_n", 128),
        m=getattr(args, "copy_m", 20),
        lr=lr,
    )
    model = NtmCopyModel(cfg)

    if optimizer_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

    print(f"  seq_width={cfg.seq_width}  seq_range=[{cfg.seq_min},{cfg.seq_max}]")
    print(f"  N={cfg.n}  M={cfg.m}  controller={cfg.controller_size}")
    print(f"  optimizer={optimizer_name}  lr={cfg.lr}  clip={clip_mode}({cfg.clip_value})")
    print(f"  iterations={cfg.iterations}")

    # Training loop: 1 sequence per iteration
    losses: list[float] = []
    for i in range(1, cfg.iterations + 1):
        input_seq, target_seq = generate_copy_sequence(
            seq_len=random.randint(cfg.seq_min, cfg.seq_max),
            seq_width=cfg.seq_width,
        )
        loss = train_ntm_copy_step(model, input_seq, target_seq, optimizer, clip_mode=clip_mode)
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

    clip_mode: Literal["value", "norm"] = getattr(args, "recall_clip", "value")
    optimizer_name: str = getattr(args, "recall_optimizer", "rmsprop")
    lr: float = getattr(args, "recall_lr", 1e-4)

    cfg = NtmRecallConfig(
        iterations=getattr(args, "recall_iters", 100000),
        n=getattr(args, "recall_n", 128),
        m=getattr(args, "recall_m", 20),
        controller_type=getattr(args, "recall_controller", "lstm"),
        min_items=getattr(args, "recall_min_items", 2),
        max_items=getattr(args, "recall_max_items", 6),
        lr=lr,
    )
    model = NtmRecallModel(cfg)

    if optimizer_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

    print(f"  seq_width={cfg.seq_width}  seq_len={cfg.seq_len}")
    print(f"  items=[{cfg.min_items},{cfg.max_items}]")
    print(f"  N={cfg.n}  M={cfg.m}  controller={cfg.controller_type}({cfg.controller_size})")
    print(f"  optimizer={optimizer_name}  lr={cfg.lr}  clip={clip_mode}({cfg.clip_value})")
    print(f"  iterations={cfg.iterations}")

    # Training loop: 1 sequence per iteration
    losses: list[float] = []
    for i in range(1, cfg.iterations + 1):
        num_items = random.randint(cfg.min_items, cfg.max_items)
        input_seq, target_seq = generate_recall_sequence(
            num_items=num_items,
            seq_len=cfg.seq_len,
            seq_width=cfg.seq_width,
        )
        loss = train_ntm_recall_step(model, input_seq, target_seq, optimizer, clip_mode=clip_mode)
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

    # Diagnostics: run instrumented forward on a 4-item sequence
    print("\n--- Diagnostics (4-item sequence) ---")
    diag_input, diag_target = generate_recall_sequence(4, cfg.seq_len, cfg.seq_width)
    timesteps = instrumented_forward_recall(model, diag_input, diag_target)
    encode_len = diag_input.shape[0]
    summary = compute_summary(timesteps, seq_len=encode_len)
    print_summary("post-training", summary)
    write_argmaxes_encode = summary.write_argmaxes[:encode_len]
    distinct_write = len(set(write_argmaxes_encode))
    print(f"  Distinct write slots (encoding): {distinct_write}")


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

    # Copy task flags
    parser.add_argument("--copy-iters", type=int, default=50000, help="Copy iterations")
    parser.add_argument("--copy-n", type=int, default=128, help="Copy memory slots")
    parser.add_argument("--copy-m", type=int, default=20, help="Copy memory width")
    parser.add_argument(
        "--copy-optimizer", choices=["rmsprop", "adam"], default="rmsprop", help="Copy optimizer"
    )
    parser.add_argument(
        "--copy-clip", choices=["value", "norm"], default="value", help="Copy clip mode"
    )
    parser.add_argument("--copy-lr", type=float, default=1e-4, help="Copy learning rate")

    # Recall task flags
    parser.add_argument("--recall-iters", type=int, default=100000, help="Recall iterations")
    parser.add_argument(
        "--recall-controller",
        choices=["lstm", "rnn"],
        default="lstm",
        help="Recall controller type",
    )
    parser.add_argument("--recall-n", type=int, default=128, help="Recall memory slots")
    parser.add_argument("--recall-m", type=int, default=20, help="Recall memory width")
    parser.add_argument(
        "--recall-optimizer",
        choices=["rmsprop", "adam"],
        default="rmsprop",
        help="Recall optimizer",
    )
    parser.add_argument(
        "--recall-clip", choices=["value", "norm"], default="value", help="Recall clip mode"
    )
    parser.add_argument("--recall-batch-size", type=int, default=1, help="Recall batch size")
    parser.add_argument(
        "--recall-output",
        choices=["read", "controller"],
        default="read",
        help="Recall output mode",
    )
    parser.add_argument("--recall-lr", type=float, default=1e-4, help="Recall learning rate")
    parser.add_argument("--recall-min-items", type=int, default=2, help="Recall min items")
    parser.add_argument("--recall-max-items", type=int, default=6, help="Recall max items")
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

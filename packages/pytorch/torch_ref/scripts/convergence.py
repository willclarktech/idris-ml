"""NTM convergence verification.

Copy task: up to 50K iterations, RMSprop, BCELoss
Recall task: up to 100K iterations, RMSprop, BCELoss

Early stopping: training halts when the average metric over the last 5000
iterations stays below a threshold for 10 consecutive checkpoints (5000 iters).

Usage:
    uv run python -m torch_ref.scripts.convergence [--task {copy,recall,both}] [--seed N]
    uv run python -m torch_ref.scripts.convergence --task recall --recall-batch-size 16
"""

import argparse
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from torch_ref.benchmark import _train_ntm_epoch
from torch_ref.data.copy_task import generate_copy_batch, generate_copy_sequence
from torch_ref.data.recall_task import generate_recall_batch, generate_recall_sequence
from torch_ref.diagnostics.ntm_diagnostics import (
    compute_summary,
    instrumented_forward_recall,
    print_summary,
)
from torch_ref.models.ntm import NtmConfig, NtmModel
from torch_ref.training.runner import get_device, set_device


def _fmt_elapsed(t0: float) -> str:
    s = int(time.monotonic() - t0)
    return f"[{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}]"


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Configuration for an NTM convergence task."""

    name: str
    ntm_cfg: NtmConfig
    batch_size: int
    iterations: int
    es_window: int = 1000
    es_patience: int = 3
    es_threshold: float = 0.01
    optimizer_name: str = "rmsprop"
    generate_batch: Callable[[], list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=lambda: lambda: []
    )
    extra_info: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------


def _train_loop(
    task: TaskConfig,
    early_stop: bool,
) -> NtmModel:
    """Run training loop with optional early stopping. Returns trained model."""
    model = NtmModel(task.ntm_cfg).to(get_device())

    if task.optimizer_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=task.ntm_cfg.lr)
    else:
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=task.ntm_cfg.lr, alpha=0.95, momentum=0.9
        )

    print(f"  N={task.ntm_cfg.n}  M={task.ntm_cfg.m}  controller={task.ntm_cfg.controller_size}")
    print(
        f"  optimizer={task.optimizer_name}  lr={task.ntm_cfg.lr}"
        f"  clip=value({task.ntm_cfg.clip_value})"
    )
    for line in task.extra_info:
        print(f"  {line}")

    print(f"  iterations={task.iterations}")
    if early_stop:
        print(
            f"  early_stop: loss<{task.es_threshold} over {task.es_window} iters,"
            f" patience={task.es_patience}"
        )

    losses: list[float] = []
    patience_count = 0
    t0 = time.monotonic()
    for i in range(1, task.iterations + 1):
        batch = task.generate_batch()
        loss = _train_ntm_epoch(model, batch, optimizer, clip_value=task.ntm_cfg.clip_value)
        losses.append(loss)

        if i % 500 == 0:
            avg_loss = sum(losses[-500:]) / len(losses[-500:])
            print(f"  {_fmt_elapsed(t0)} iter {i:6d}: loss={avg_loss:.6f}")

            if early_stop and i >= task.es_window:
                window_avg = sum(losses[-task.es_window :]) / task.es_window
                if window_avg < task.es_threshold:
                    patience_count += 1
                    if patience_count >= task.es_patience:
                        print(
                            f"  {_fmt_elapsed(t0)} ** early stop at iter {i}"
                            f" (avg loss={window_avg:.6f})"
                        )
                        break
                else:
                    patience_count = 0

    return model


# ---------------------------------------------------------------------------
# Copy task
# ---------------------------------------------------------------------------

COPY_SEQ_WIDTH = 8


def _eval_copy(model: NtmModel) -> None:
    """Evaluate copy task on test lengths."""
    print("\n--- Evaluation ---")
    model.eval()
    for test_len in [5, 10, 20]:
        correct_bits = 0
        total_bits = 0
        for _ in range(10):
            input_seq, target_seq = generate_copy_sequence(test_len, COPY_SEQ_WIDTH)
            with torch.no_grad():
                model.reset_state()
                for t in range(input_seq.shape[0]):
                    model(input_seq[t])
                outputs = []
                for _ in range(test_len):
                    out = model(torch.zeros(input_seq.shape[1], device=get_device()))
                    outputs.append(out)
                pred = torch.sigmoid(torch.stack(outputs))
                pred_bits = (pred > 0.5).float()
                correct_bits += (pred_bits == target_seq).sum().item()
                total_bits += target_seq.numel()

        accuracy = correct_bits / total_bits if total_bits > 0 else 0
        print(f"  Length {test_len:2d}: bit accuracy = {accuracy:.1%}")


def run_copy(args: argparse.Namespace) -> None:
    """Train NTM on copy task."""
    print("=" * 60)
    print("NTM Copy Task Convergence")
    print("=" * 60)

    lr: float = getattr(args, "copy_lr", 1e-4)
    iterations: int = getattr(args, "copy_iters", 50000)
    n: int = getattr(args, "copy_n", 128)
    m: int = getattr(args, "copy_m", 20)
    batch_size: int = getattr(args, "copy_batch_size", 16)

    cfg = NtmConfig(input_width=COPY_SEQ_WIDTH + 1, output_width=COPY_SEQ_WIDTH, n=n, m=m, lr=lr)

    task = TaskConfig(
        name="copy",
        ntm_cfg=cfg,
        batch_size=batch_size,
        iterations=iterations,
        optimizer_name=getattr(args, "copy_optimizer", "rmsprop"),
        generate_batch=lambda: generate_copy_batch(
            batch_size, seq_min=1, seq_max=20, seq_width=COPY_SEQ_WIDTH
        ),
        extra_info=[
            f"seq_width={COPY_SEQ_WIDTH}  seq_range=[1,20]  batch_size={batch_size}",
        ],
    )

    early_stop = not getattr(args, "no_early_stop", False)
    model = _train_loop(task, early_stop)
    _eval_copy(model)


# ---------------------------------------------------------------------------
# Recall task
# ---------------------------------------------------------------------------

RECALL_SEQ_WIDTH = 6
RECALL_SEQ_LEN = 3


def _eval_recall(model: NtmModel) -> None:
    """Evaluate recall task on different item counts."""
    print("\n--- Evaluation ---")
    model.eval()
    for test_items in [2, 3, 4, 5, 6]:
        correct_bits = 0
        total_bits = 0
        total_bit_errors = 0.0
        num_trials = 10
        for _ in range(num_trials):
            input_seq, target_seq = generate_recall_sequence(
                test_items, RECALL_SEQ_LEN, RECALL_SEQ_WIDTH
            )
            with torch.no_grad():
                model.reset_state()
                seq_len = target_seq.shape[0]

                for t in range(input_seq.shape[0]):
                    model(input_seq[t])

                zero_input = torch.zeros(input_seq.shape[1], device=get_device())
                outputs = []
                for _ in range(seq_len):
                    out = model(zero_input)
                    outputs.append(out)

                pred = torch.sigmoid(torch.stack(outputs))
                pred_bits = (pred >= 0.5).float()
                correct_bits += (pred_bits == target_seq).sum().item()
                total_bits += target_seq.numel()
                total_bit_errors += torch.sum(torch.abs(pred_bits - target_seq)).item()

        accuracy = correct_bits / total_bits if total_bits > 0 else 0
        avg_bit_err = total_bit_errors / num_trials
        bits_per_seq = total_bits / num_trials
        print(
            f"  {test_items} items: {accuracy:.1%} "
            f"({avg_bit_err:.1f}/{bits_per_seq:.0f} bit errors/seq)"
        )

    # Diagnostics: run instrumented forward on a 4-item sequence
    print("\n--- Diagnostics (4-item sequence) ---")
    diag_input, diag_target = generate_recall_sequence(4, RECALL_SEQ_LEN, RECALL_SEQ_WIDTH)
    timesteps = instrumented_forward_recall(model, diag_input, diag_target)
    encode_len = diag_input.shape[0]
    summary = compute_summary(timesteps, seq_len=encode_len)
    print_summary("post-training", summary)
    write_argmaxes_encode = summary.write_argmaxes[:encode_len]
    distinct_write = len(set(write_argmaxes_encode))
    print(f"  Distinct write slots (encoding): {distinct_write}")


def run_recall(args: argparse.Namespace) -> None:
    """Train NTM on associative recall task."""
    print("=" * 60)
    print("NTM Associative Recall Convergence")
    print("=" * 60)

    lr: float = getattr(args, "recall_lr", 1e-4)
    iterations: int = getattr(args, "recall_iters", 100000)
    n: int = getattr(args, "recall_n", 128)
    m: int = getattr(args, "recall_m", 20)
    min_items: int = getattr(args, "recall_min_items", 2)
    max_items: int = getattr(args, "recall_max_items", 6)
    batch_size: int = getattr(args, "recall_batch_size", 1)

    cfg = NtmConfig(
        input_width=RECALL_SEQ_WIDTH + 2, output_width=RECALL_SEQ_WIDTH, n=n, m=m, lr=lr
    )

    task = TaskConfig(
        name="recall",
        ntm_cfg=cfg,
        batch_size=batch_size,
        iterations=iterations,
        optimizer_name=getattr(args, "recall_optimizer", "rmsprop"),
        generate_batch=lambda: generate_recall_batch(
            batch_size, min_items, max_items, RECALL_SEQ_LEN, RECALL_SEQ_WIDTH
        ),
        extra_info=[
            f"seq_width={RECALL_SEQ_WIDTH}  seq_len={RECALL_SEQ_LEN}  batch_size={batch_size}",
            f"items=[{min_items},{max_items}]",
        ],
    )

    early_stop = not getattr(args, "no_early_stop", False)
    model = _train_loop(task, early_stop)
    _eval_recall(model)


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
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")

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
    parser.add_argument(
        "--copy-batch-size", type=int, default=1, help="Copy batch size (default: 1)"
    )

    # Recall task flags
    parser.add_argument("--recall-iters", type=int, default=100000, help="Recall iterations")
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
    parser.add_argument("--recall-lr", type=float, default=1e-4, help="Recall learning rate")
    parser.add_argument("--recall-min-items", type=int, default=2, help="Recall min items")
    parser.add_argument("--recall-max-items", type=int, default=6, help="Recall max items")
    parser.add_argument(
        "--recall-batch-size",
        type=int,
        default=1,
        help="Recall batch size (default: 1, matching reference implementations)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for tensor ops (default: cpu)",
    )
    args = parser.parse_args()

    set_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.task in ("copy", "both"):
        run_copy(args)
        print()

    if args.task in ("recall", "both"):
        run_recall(args)


if __name__ == "__main__":
    main()

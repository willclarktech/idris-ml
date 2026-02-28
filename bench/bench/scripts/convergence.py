"""NTM convergence verification with content-addressing diagnostics.

Trains NTM on copy and/or associative recall tasks using curriculum learning,
then runs diagnostics to verify addressing behavior.

Usage:
    uv run python -m bench.scripts.convergence [--task {copy,recall,both}] [--seed N]
        [--copy-epochs N] [--recall-epochs N] [--verbose]
        [--recall-controller {rnn,lstm}] [--recall-n N]
"""

import argparse
import math
import random
from functools import partial

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from bench.data.copy_task import generate_copy_batch
from bench.data.recall_task import generate_recall_batch
from bench.diagnostics.ntm_diagnostics import (
    NtmSummary,
    compute_content_match,
    compute_summary,
    instrumented_forward,
    print_addr_grid,
    print_comparison,
    print_summary,
)
from bench.models.ntm_copy import NtmCopyConfig, NtmCopyModel
from bench.models.ntm_recall import NtmRecallConfig, NtmRecallModel, train_ntm_recall_step
from bench.training.curriculum import Stage, run_curriculum
from bench.training.losses import nll_loss, weighted_nll_loss

# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------


def sequence_accuracy(
    model: torch.nn.Module,
    inputs: list[Tensor],
    targets: list[Tensor],
    output_only: bool = False,
    seq_len: int | None = None,
) -> float:
    """Compute accuracy as fraction of timesteps with matching argmax.

    If output_only=True, only count timesteps in the output phase (t >= seq_len).
    """
    model.eval()  # type: ignore[attr-defined]
    model.reset_state()  # type: ignore[attr-defined]
    correct = 0
    total = 0
    start_t = seq_len if (output_only and seq_len is not None) else 0

    with torch.no_grad():
        for t, (x, y) in enumerate(zip(inputs, targets, strict=True)):
            pred = model(x)  # type: ignore[operator]
            if t >= start_t:
                if pred.argmax() == y.argmax():
                    correct += 1
                total += 1

    model.train()  # type: ignore[attr-defined]
    return correct / total if total > 0 else 0.0


def recall_accuracy(
    model: torch.nn.Module,
    inputs: list[Tensor],
    targets: list[Tensor],
) -> float:
    """Accuracy counting only query-response timesteps (target != blank)."""
    model.eval()  # type: ignore[attr-defined]
    model.reset_state()  # type: ignore[attr-defined]
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in zip(inputs, targets, strict=True):
            pred = model(x)  # type: ignore[operator]
            if y.argmax() != 0:  # non-blank target
                if pred.argmax() == y.argmax():
                    correct += 1
                total += 1

    model.train()  # type: ignore[attr-defined]
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Copy task
# ---------------------------------------------------------------------------


def run_copy(args: argparse.Namespace) -> None:
    """Train and diagnose NTM copy task."""
    print("=" * 60)
    print("NTM Copy Task Convergence")
    print("=" * 60)

    cfg = NtmCopyConfig(epochs=args.copy_epochs)
    model = NtmCopyModel(cfg)
    w = cfg.w

    # Curriculum stages
    stages = [
        Stage(
            "Stage 1 (len 1-3)",
            0.15,
            partial(generate_copy_batch, cfg.batch_size, 1, 3, w),
        ),
        Stage(
            "Stage 2 (len 1-5)",
            0.10,
            partial(generate_copy_batch, cfg.batch_size, 1, 5, w),
        ),
        Stage(
            "Stage 3 (len 1-8)",
            0.0,
            partial(generate_copy_batch, cfg.batch_size, 1, 8, w),
        ),
    ]

    # One-cycle LR schedule
    base_lr = cfg.lr
    peak_ratio = 25.0
    pct_start = 0.25

    def schedule_fn(epoch: int) -> float:
        warmup = int(pct_start * cfg.epochs)
        if epoch < warmup:
            lr_start = base_lr / peak_ratio
            return lr_start + (base_lr - lr_start) * epoch / max(warmup, 1)
        else:
            lr_end = base_lr / cfg.div_final
            progress = (epoch - warmup) / max(cfg.epochs - warmup, 1)
            return lr_end + (base_lr - lr_end) * 0.5 * (1 + math.cos(math.pi * progress))

    def optimizer_factory(m: torch.nn.Module, lr: float) -> torch.optim.Adam:
        return torch.optim.Adam(m.parameters(), lr=lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)

    def train_step(
        m: NtmCopyModel,
        data: list[tuple[list[Tensor], list[Tensor]]],
        loss_fn: object,
        opt: torch.optim.Optimizer,
    ) -> float:
        opt.zero_grad()
        total_loss = torch.tensor(0.0)
        for xs, ys in data:
            m.reset_state()
            seq_loss = torch.tensor(0.0)
            for x, y in zip(xs, ys, strict=True):
                pred = m(x)
                seq_loss = seq_loss + nll_loss(pred, y)
            total_loss = total_loss + seq_loss / len(xs)
        loss = total_loss / len(data)
        loss.backward()
        clip_grad_norm_(m.parameters(), cfg.max_norm)
        opt.step()
        m.project_addressing()
        return loss.item()

    # Train
    epochs_done, final_loss = run_curriculum(
        model=model,
        loss_fn=nll_loss,
        stages=stages,
        total_epochs=cfg.epochs,
        patience=cfg.patience,
        chunk_size=cfg.chunk_size,
        optimizer_factory=optimizer_factory,
        schedule_fn=schedule_fn,
        train_step_fn=train_step,
    )
    print(f"\nTraining complete: {epochs_done} epochs, final loss={final_loss:.6f}")

    # Evaluate accuracy
    print("\n--- Accuracy ---")
    model.eval()
    for test_len in [3, 5, 8]:
        test_data = generate_copy_batch(10, test_len, test_len, w)
        accs = []
        for xs, ys in test_data:
            acc = sequence_accuracy(model, xs, ys, output_only=True, seq_len=test_len)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)
        print(f"  Length {test_len}: {avg_acc:.1%}")

    # Diagnostics
    print("\n--- Diagnostics ---")
    train_summaries: list[NtmSummary] = []
    test_summaries: list[NtmSummary] = []

    for test_len in [3, 5, 8]:
        # Train example
        train_data = generate_copy_batch(1, test_len, test_len, w)
        xs, ys = train_data[0]
        ts = instrumented_forward(model, xs, ys)
        s = compute_summary(ts, test_len)
        train_summaries.append(s)
        print_summary(f"Copy len={test_len} (train)", s)
        print_addr_grid(s)

        # Test example (fresh)
        test_data = generate_copy_batch(1, test_len, test_len, w)
        xs, ys = test_data[0]
        ts = instrumented_forward(model, xs, ys)
        s = compute_summary(ts, test_len)
        test_summaries.append(s)
        print_summary(f"Copy len={test_len} (test)", s)
        print_addr_grid(s)
        print()

    # Averaged comparison
    if train_summaries and test_summaries:
        avg_train = _avg_summaries(train_summaries)
        avg_test = _avg_summaries(test_summaries)
        if avg_train and avg_test:
            print_comparison(avg_train, avg_test)


# ---------------------------------------------------------------------------
# Recall task
# ---------------------------------------------------------------------------


def run_recall(args: argparse.Namespace) -> None:
    """Train and diagnose NTM associative recall task."""
    controller = getattr(args, "recall_controller", "lstm")
    n = getattr(args, "recall_n", 128)
    optimizer_type = getattr(args, "recall_optimizer", "rmsprop")
    clip_mode = getattr(args, "recall_clip", "value")
    clip_value = getattr(args, "recall_clip_value", 10.0)
    batch_size = getattr(args, "recall_batch_size", 1)
    use_curriculum = getattr(args, "recall_curriculum", False)
    recall_k = getattr(args, "recall_k", 2)

    print("=" * 60)
    print("NTM Associative Recall Convergence")
    print(f"  controller={controller}  N={n}  optimizer={optimizer_type}")
    print(f"  clip={clip_mode}  batch_size={batch_size}  curriculum={use_curriculum}")
    print("=" * 60)

    cfg = NtmRecallConfig(
        epochs=args.recall_epochs,
        patience=1000,
        controller=controller,
        n=n,
        optimizer=optimizer_type,
        clip_mode=clip_mode,
        clip_value=clip_value,
        batch_size=batch_size,
    )
    model = NtmRecallModel(cfg)
    w = cfg.w

    if use_curriculum:
        # 3-stage curriculum (fixed K per stage, capped at K=3)
        stages = [
            Stage(
                "Stage 1 (K=1)",
                0.15,
                partial(generate_recall_batch, cfg.batch_size, 1, 1, w),
            ),
            Stage(
                "Stage 2 (K=2)",
                0.12,
                partial(generate_recall_batch, cfg.batch_size, 2, 2, w),
            ),
            Stage(
                "Stage 3 (K=3)",
                0.0,
                partial(generate_recall_batch, cfg.batch_size, 3, 3, w),
            ),
        ]
    else:
        # Single stage, direct K training (reference implementations use no curriculum)
        stages = [
            Stage(
                f"Direct K={recall_k}",
                0.0,
                partial(generate_recall_batch, cfg.batch_size, recall_k, recall_k, w),
            ),
        ]

    # Schedule: constant LR for RMSprop (reference), one-cycle for Adam
    base_lr = cfg.lr
    if optimizer_type == "rmsprop":

        def schedule_fn(epoch: int) -> float:
            return base_lr
    else:
        peak_ratio = 25.0
        pct_start = 0.25

        def schedule_fn(epoch: int) -> float:
            warmup = int(pct_start * cfg.epochs)
            if epoch < warmup:
                lr_start = base_lr / peak_ratio
                return lr_start + (base_lr - lr_start) * epoch / max(warmup, 1)
            else:
                lr_end = base_lr / cfg.div_final
                progress = (epoch - warmup) / max(cfg.epochs - warmup, 1)
                return lr_end + (base_lr - lr_end) * 0.5 * (1 + math.cos(math.pi * progress))

    def optimizer_factory(m: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        if optimizer_type == "rmsprop":
            return torch.optim.RMSprop(m.parameters(), lr=lr, alpha=0.95, momentum=0.9)
        return torch.optim.Adam(m.parameters(), lr=lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)

    def train_step(
        m: NtmRecallModel,
        data: list[tuple[list[Tensor], list[Tensor]]],
        loss_fn: object,
        opt: torch.optim.Optimizer,
    ) -> float:
        return train_ntm_recall_step(m, data, loss_fn, opt)

    # Train
    epochs_done, final_loss = run_curriculum(
        model=model,
        loss_fn=weighted_nll_loss,
        stages=stages,
        total_epochs=cfg.epochs,
        patience=cfg.patience,
        chunk_size=cfg.chunk_size,
        optimizer_factory=optimizer_factory,
        schedule_fn=schedule_fn,
        train_step_fn=train_step,
    )
    print(f"\nTraining complete: {epochs_done} epochs, final loss={final_loss:.6f}")

    # Evaluate accuracy
    print("\n--- Accuracy ---")
    model.eval()
    for test_k in [1, 2, 3]:
        test_data = generate_recall_batch(10, test_k, test_k, w)
        accs = []
        for xs, ys in test_data:
            acc = recall_accuracy(model, xs, ys)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)
        print(f"  K={test_k}: {avg_acc:.1%}")

    # Diagnostics
    print("\n--- Diagnostics ---")
    train_summaries: list[NtmSummary] = []
    test_summaries: list[NtmSummary] = []

    for test_k in [2, 3]:
        test_data = generate_recall_batch(2, test_k, test_k, w)
        for seq_idx, (xs, ys) in enumerate(test_data):
            ts = instrumented_forward(model, xs, ys)

            # Recall: seq_len = 2K (store phase) + 1 (delimiter)
            # But for summary: input phase = 2K+1, output phase = 2K
            store_len = 2 * test_k + 1  # store + delimiter
            s = compute_summary(ts, store_len)

            # Content match analysis
            # Store write slots: argmax at key timesteps (t=0,2,4,...)
            key_steps = range(0, 2 * test_k, 2)
            store_write_slots = [int(ts[t].write_addr.argmax().item()) for t in key_steps]
            # Query response timesteps: after delimiter, every other step (blank input = response)
            # Query phase starts at 2K+1, responses at 2K+2, 2K+4, ...
            query_response_indices = list(range(2 * test_k + 2, len(ts), 2))

            # Map query to store: we need to figure out which key was queried
            # Query keys are at positions 2K+1, 2K+3, ... in the input
            query_key_indices = list(range(2 * test_k + 1, len(ts), 2))
            # The store keys are at positions 0, 2, 4, ... with symbols
            store_keys = [int(xs[t].argmax().item()) for t in range(0, 2 * test_k, 2)]
            query_to_store: list[int] = []
            for qi in query_key_indices:
                q_sym = int(xs[qi].argmax().item())
                # Find which store index has this key
                store_idx = store_keys.index(q_sym) if q_sym in store_keys else 0
                query_to_store.append(store_idx)

            # Trim to match response indices length
            query_to_store = query_to_store[: len(query_response_indices)]
            if query_response_indices and store_write_slots:
                s.content_match_rate = compute_content_match(
                    ts, store_write_slots, query_response_indices, query_to_store
                )

            label = f"Recall K={test_k} seq#{seq_idx}"
            print_summary(label, s)
            print_addr_grid(s)

            # Print query-phase g values
            print("  Query-phase detail:")
            for qi in query_response_indices:
                if qi < len(ts):
                    t = ts[qi]
                    ca = int(t.content_read_weights.argmax().item())
                    ra = int(t.read_addr.argmax().item())
                    target = int(ys[qi].argmax().item())
                    print(
                        f"    t{qi}: read_g={t.read_g:.3f}"
                        f"  content_argmax={ca}"
                        f"  actual_read={ra}"
                        f"  target={target}"
                    )
            print()

            if seq_idx == 0:
                train_summaries.append(s)
            else:
                test_summaries.append(s)

    # Averaged comparison
    if train_summaries and test_summaries:
        avg_train = _avg_summaries(train_summaries)
        avg_test = _avg_summaries(test_summaries)
        if avg_train and avg_test:
            print_comparison(avg_train, avg_test)


# ---------------------------------------------------------------------------
# Summary averaging helper
# ---------------------------------------------------------------------------


def _avg_summaries(summaries: list[NtmSummary]) -> NtmSummary | None:
    if not summaries:
        return None
    n = len(summaries)

    def avg_field(getter: object) -> float:
        return sum(getter(s) for s in summaries) / n  # type: ignore[operator]

    cm_vals = [s.content_match_rate for s in summaries if s.content_match_rate is not None]
    avg_cm = sum(cm_vals) / len(cm_vals) if cm_vals else None

    return NtmSummary(
        write_g_input=avg_field(lambda s: s.write_g_input),
        write_g_output=avg_field(lambda s: s.write_g_output),
        read_g_input=avg_field(lambda s: s.read_g_input),
        read_g_output=avg_field(lambda s: s.read_g_output),
        avg_write_beta=avg_field(lambda s: s.avg_write_beta),
        avg_read_beta=avg_field(lambda s: s.avg_read_beta),
        avg_write_gamma=avg_field(lambda s: s.avg_write_gamma),
        avg_read_gamma=avg_field(lambda s: s.avg_read_gamma),
        write_addr_entropy=avg_field(lambda s: s.write_addr_entropy),
        read_addr_entropy=avg_field(lambda s: s.read_addr_entropy),
        write_addr_peak_mass=avg_field(lambda s: s.write_addr_peak_mass),
        read_addr_peak_mass=avg_field(lambda s: s.read_addr_peak_mass),
        write_monotonic=all(s.write_monotonic for s in summaries),
        read_monotonic=all(s.read_monotonic for s in summaries),
        write_argmaxes=summaries[0].write_argmaxes,
        read_argmaxes=summaries[0].read_argmaxes,
        slots_used=summaries[0].slots_used,
        num_slots=summaries[0].num_slots,
        seq_len=summaries[0].seq_len,
        content_match_rate=avg_cm,
    )


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
    parser.add_argument("--copy-epochs", type=int, default=6000, help="Max epochs for copy task")
    parser.add_argument(
        "--recall-epochs", type=int, default=15000, help="Max epochs for recall task"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-timestep raw state")
    parser.add_argument(
        "--recall-controller",
        choices=["rnn", "lstm"],
        default="lstm",
        help="Recall controller type (default: lstm)",
    )
    parser.add_argument(
        "--recall-n", type=int, default=128, help="Recall memory slots (default: 128)"
    )
    parser.add_argument(
        "--recall-optimizer",
        choices=["adam", "rmsprop"],
        default="rmsprop",
        help="Recall optimizer (default: rmsprop)",
    )
    parser.add_argument(
        "--recall-clip",
        choices=["norm", "value"],
        default="value",
        help="Gradient clipping mode (default: value)",
    )
    parser.add_argument(
        "--recall-clip-value", type=float, default=10.0, help="Value clip bound (default: 10.0)"
    )
    parser.add_argument(
        "--recall-batch-size", type=int, default=1, help="Recall batch size (default: 1)"
    )
    parser.add_argument(
        "--recall-curriculum", action="store_true", help="Use 3-stage curriculum (default: off)"
    )
    parser.add_argument(
        "--recall-k", type=int, default=2, help="K for direct training (default: 2)"
    )
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

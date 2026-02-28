"""Multi-stage curriculum training matching idris-ml's Curriculum.idr.

Periodic data regeneration with two-level early stopping:
- Within-stage patience (stale count)
- Stage advancement on loss threshold
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

MIN_DELTA = 0.001


@dataclass
class Stage:
    label: str
    threshold: float  # loss below this advances to next stage (0.0 = never auto-advance)
    generate: Callable[[], list[tuple[list[Tensor], list[Tensor]]]]


def _one_cycle_lr(
    base_lr: float,
    peak_ratio: float,
    div_final: float,
    pct_start: float,
    total_epochs: int,
    epoch: int,
) -> float:
    """One-cycle learning rate schedule matching Schedule.idr oneCycle."""
    warmup_epochs = int(pct_start * total_epochs)
    if epoch < warmup_epochs:
        lr_start = base_lr / peak_ratio
        return lr_start + (base_lr - lr_start) * epoch / max(warmup_epochs, 1)
    else:
        lr_end = base_lr / div_final
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return lr_end + (base_lr - lr_end) * 0.5 * (1 + math.cos(math.pi * progress))


def run_curriculum(
    model: Any,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    stages: list[Stage],
    total_epochs: int,
    patience: int,
    chunk_size: int,
    optimizer_factory: Callable[[Any, float], torch.optim.Optimizer],
    schedule_fn: Callable[[int], float],
    post_step_fn: Callable[[], None] | None = None,
    train_step_fn: Callable[
        [Any, list[tuple[list[Tensor], list[Tensor]]], Callable, torch.optim.Optimizer],
        float,
    ]
    | None = None,
) -> tuple[int, float]:
    """Run multi-stage curriculum training.

    Matches Curriculum.idr runCurriculum.
    Returns: (total_epochs_done, final_loss)
    """
    done = 0
    budget = total_epochs
    best_loss = float("inf")
    stale_count = 0

    # Create optimizer once — preserve internal state (RMSprop running averages, Adam moments)
    initial_lr = schedule_fn(0)
    optimizer = optimizer_factory(model, initial_lr)

    for stage_idx, stage in enumerate(stages):
        print(f"\n{stage.label}")
        advanced = False

        while budget > 0:
            # Generate fresh data
            data = stage.generate()
            chunk = min(chunk_size, budget)

            # Train chunk
            loss_val = best_loss
            for _ in range(chunk):
                lr = schedule_fn(done)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if train_step_fn is not None:
                    loss_val = train_step_fn(model, data, loss_fn, optimizer)
                else:
                    raise ValueError("train_step_fn must be provided")

                if post_step_fn is not None:
                    post_step_fn()

                done += 1
                budget -= 1

                improved = loss_val < best_loss - MIN_DELTA
                if improved:
                    best_loss = loss_val
                    stale_count = 0
                else:
                    stale_count += 1

            print(f"  {done}:\t{loss_val:.6f}")

            # Check stage advancement
            if stage.threshold > 0.0 and loss_val < stage.threshold:
                print(f"  -> Advancing (loss {loss_val:.6f} < {stage.threshold})")
                advanced = True
                break

            if patience > 0 and stale_count >= patience:
                print(f"  Early stop at epoch {done} (patience={patience})")
                break

            if loss_val != loss_val:  # NaN check
                print(f"  Diverged (NaN) at epoch {done}")
                return done, loss_val

        if not advanced and stage_idx < len(stages) - 1:
            # Patience exhausted, stop training
            break

    return done, best_loss

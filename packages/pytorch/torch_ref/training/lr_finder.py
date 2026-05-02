"""LR-range test (Smith 2017, popularized by fastai's `lr_find`).

Mirror of `packages/idris-ml/src/Hpo/LrFinder.idr` — same algorithm, same
heuristic, same stdout format. Both backends should recommend an LR
within ~2× of each other on a fixed-seed synthetic problem.

Trains for `num_iters` iterations with LR sweeping log-uniformly from
`lr_min` (default 1e-7) to `lr_max` (default 10), records the EMA-smoothed
loss at each LR, and recommends an LR using fastai's heuristic: take the
LR at the steepest negative slope of smoothed loss vs log(LR), divided
by `recommend_div` (default 10) for a safety margin.

Single-seed, single-batch by design — `lr_find` is a quick screening
tool; multi-seed validation belongs to whatever follow-up training run
consumes the recommendation.

Note: this mutates the model and the optimizer's LR. The caller should
either save+restore optimizer state or construct a fresh optimizer for
the actual training run after the recommendation.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class LrFindConfig:
    """LR-range-test configuration. Defaults match fastai."""

    lr_min: float = 1e-7
    lr_max: float = 10.0
    num_iters: int = 100
    smooth_beta: float = 0.98
    diverge_factor: float = 4.0
    recommend_div: float = 10.0


@dataclass
class LrFindResult:
    """Result of an LR-range test.

    `points` is a list of (lr, smoothed_loss) pairs in iteration order.
    Iterations beyond divergence are omitted.
    """

    points: list[tuple[float, float]]
    recommended_lr: float


def sweep_lr(lr_min: float, lr_max: float, n: int, i: int) -> float:
    """LR at iteration `i` of `n` total in log-space.

    `sweep_lr(_, _, n, 0) == lr_min`,
    `sweep_lr(_, _, n, n-1) == lr_max`.
    """
    if n <= 1:
        return lr_min
    frac = i / (n - 1)
    return lr_min * math.exp(frac * math.log(lr_max / lr_min))


def _set_lr(optimizer: Any, lr: float) -> None:
    """Update the optimizer's base LR across all param groups.

    PyTorch optimizers store LR per param group; we mirror Idris'
    `optimizer_set_lr` semantics by writing the same value to every group.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr


def _slopes(curve: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Adjacent slopes: (lr_i, smoothed_{i+1} - smoothed_i). Last point
    has no successor and is dropped."""
    if len(curve) < 2:
        return []
    return [(curve[i][0], curve[i + 1][1] - curve[i][1]) for i in range(len(curve) - 1)]


def _steepest_descent(slopes: list[tuple[float, float]]) -> float:
    """LR at the most-negative slope. Falls back to the first LR if all
    slopes are non-negative (loss never decreased — unusual)."""
    if not slopes:
        return 0.0
    best_lr, best_slope = slopes[0]
    for lr, s in slopes[1:]:
        if s < best_slope:
            best_lr, best_slope = lr, s
    return best_lr


def recommend_from_curve(recommend_div: float, curve: list[tuple[float, float]]) -> float:
    """Recommend an LR from the swept curve via fastai's heuristic."""
    return _steepest_descent(_slopes(curve)) / recommend_div


def lr_find(
    config: LrFindConfig,
    epoch_fn: Callable[[], float],
    optimizer: Any,
) -> LrFindResult:
    """Run the LR range test.

    Args:
        config: LrFindConfig.
        epoch_fn: callable that runs ONE forward+backward+step and returns
            the scalar loss. The optimizer's LR is updated externally each
            iteration; epoch_fn just consumes the current state.
        optimizer: PyTorch optimizer instance whose `lr` will be swept.

    Returns:
        LrFindResult with the (lr, smoothed_loss) curve and the
        recommended LR.

    Stdout output: one line per iteration as
        `iter\\t<i>\\tlr\\t<lr>\\tloss\\t<loss>\\tsmoothed\\t<smoothed>`
    followed by a final
        `RECOMMENDED_LR=<value>`
    line.
    """
    print(
        f"lr_find: sweeping LR from {config.lr_min} to {config.lr_max}"
        f" over {config.num_iters} iters"
    )
    t_start = time.monotonic()

    points: list[tuple[float, float]] = []
    prev_smoothed = 0.0
    min_smoothed = math.inf

    for i in range(config.num_iters):
        lr = sweep_lr(config.lr_min, config.lr_max, config.num_iters, i)
        _set_lr(optimizer, lr)
        loss = epoch_fn()

        beta = config.smooth_beta
        avg = beta * prev_smoothed + (1.0 - beta) * loss
        # bias-corrected EMA
        corrected = avg / (1.0 - beta ** (i + 1))
        prev_smoothed = avg
        if corrected < min_smoothed:
            min_smoothed = corrected
        points.append((lr, corrected))

        print(
            f"  iter\t{i}\tlr\t{lr}\tloss\t{loss}\tsmoothed\t{corrected}"
        )

        if i > 0 and corrected > config.diverge_factor * min_smoothed:
            print(
                f"  (diverged at iter {i}, smoothed={corrected}"
                f" > {config.diverge_factor} * min={min_smoothed})"
            )
            break

    elapsed = time.monotonic() - t_start
    print(f"lr_find done in {elapsed:.2f}s")

    rec = recommend_from_curve(config.recommend_div, points)
    print(f"RECOMMENDED_LR={rec}")

    return LrFindResult(points=points, recommended_lr=rec)

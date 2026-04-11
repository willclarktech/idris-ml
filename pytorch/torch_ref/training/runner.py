"""Unified training runner matching Idris runTraining output format.

Handles epoch loop, progress logging, early stopping, timing, and
result formatting. Output is identical to the Idris Train.idr runner.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training configuration mirroring Idris TrainConfig."""

    total_epochs: int = 1000
    log_every: int = 100
    patience: int = 0  # 0 = no early stopping
    min_delta: float = 0.001


def _format_elapsed(start: float) -> str:
    """Format elapsed time as [HH:MM:SS]."""
    total = int(time.monotonic() - start)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"[{hh:02d}:{mm:02d}:{ss:02d}]"


def _format_duration(total_sec: int) -> str:
    """Format duration as 'Xm Ys' or 'Xh Ym'."""
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_metrics(metrics: list[tuple[str, str]]) -> str:
    """Format metrics as tab-separated key=value pairs."""
    return "".join(f"\t{k}={v}" for k, v in metrics)


def format_result(kvs: list[tuple[str, str]]) -> str:
    """Format a machine-readable RESULT line from key-value pairs."""
    return "RESULT" + "".join(f"\t{k}={v}" for k, v in kvs)


def run_training(
    epoch_fn: Callable[[], float],
    config: TrainConfig,
    metrics_fn: Callable[[], list[tuple[str, str]]] | None = None,
) -> tuple[int, float]:
    """Run training loop with Idris-matching output format.

    Args:
        epoch_fn: Callable that runs one epoch and returns loss.
        config: Training configuration.
        metrics_fn: Optional callable returning extra metrics for logging.

    Returns:
        (epochs_completed, final_loss)
    """
    t_start = time.monotonic()
    print("Training...")

    best_loss = math.inf
    stale = 0
    final_loss = 0.0
    epochs_done = 0

    for ep in range(config.total_epochs):
        loss = epoch_fn()
        epochs_done = ep + 1

        # Log progress
        if config.log_every > 0 and ep % config.log_every == 0:
            extra = metrics_fn() if metrics_fn else []
            print(f"  {_format_elapsed(t_start)} {ep}\tloss={loss}{_format_metrics(extra)}")

        # NaN detection
        if math.isnan(loss):
            print(f"  {_format_elapsed(t_start)} Diverged (NaN) at epoch {ep}")
            return epochs_done, loss

        final_loss = loss

        # Patience-based early stopping
        if config.patience > 0:
            if loss < best_loss - config.min_delta:
                best_loss = loss
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                print(
                    f"  {_format_elapsed(t_start)} Early stop at epoch {epochs_done}"
                    f" (patience={config.patience})"
                )
                break

    total_sec = int(time.monotonic() - t_start)
    dur = _format_duration(total_sec)
    ms_per = (total_sec * 1000) // epochs_done if epochs_done > 0 else 0
    print(f"Completed in {dur} ({epochs_done} epochs, {ms_per}ms/epoch)")

    return epochs_done, final_loss

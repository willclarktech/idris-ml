"""Unified training runner matching Idris runTraining output format.

Handles epoch loop, progress logging, early stopping, timing, and
result formatting. Output is identical to the Idris Train.idr runner.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

_PROC = psutil.Process()
_PEAK_MB = 0


@dataclass
class TrainConfig:
    """Training configuration mirroring Idris TrainConfig."""

    total_epochs: int = 1000
    log_every: int = 100
    # Patience-based early stopping (0 = disabled)
    patience: int = 0
    min_delta: float = 0.001
    # Windowed-average early stopping (threshold=0 = disabled)
    windowed_threshold: float = 0.0
    windowed_window: int = 1000
    windowed_patience: int = 3
    # Windowed-percentile early stopping (mirrors Idris `WindowedPercentile`).
    # When > 0, replaces the windowed-mean check with a percentile of the
    # last `windowed_window` raw per-epoch losses. Required for bimodal-loss
    # tasks (variable-length-sequence training) where the mean over a
    # window stays high even after convergence on easy sequences.
    windowed_percentile: float = 0.0
    # Pre-epoch hook (mirrors Idris `beforeEpoch : Nat -> IO ()`). Receives
    # the current epoch index. Use this to apply LR schedules manually if
    # you don't have a `torch.optim.lr_scheduler` instance handy.
    before_epoch: Callable[[int], None] = field(default=lambda _ep: None)
    # Optional `torch.optim.lr_scheduler._LRScheduler` (or any object with
    # a `.step()` method). Stepped once per epoch after `epoch_fn` runs,
    # matching PyTorch convention.
    lr_scheduler: Any | None = None


def format_elapsed(start: float) -> str:
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


def mem_suffix() -> str:
    """Format `\\tpeak=NMB\\tcur=NMB` for the unified per-epoch log line.

    Updates a module-level peak watermark on every call so the value reflects
    the high-water mark across the whole training run. Mirrors the Idris
    `Tensor.getRssMB` / `getCurrentRssMB` pair used by `Train.logEpoch`.
    """
    global _PEAK_MB
    cur_mb = _PROC.memory_info().rss // (1024 * 1024)
    if cur_mb > _PEAK_MB:
        _PEAK_MB = cur_mb
    return f"\tpeak={_PEAK_MB}MB\tcur={cur_mb}MB"


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

    # Windowed early-stopping state. Two flavours:
    #   - mean-of-100-epoch-chunks (existing, used by most refs)
    #   - percentile-of-raw-losses (new, mirrors Idris WindowedPercentile)
    use_windowed = config.windowed_threshold > 0
    use_percentile = use_windowed and config.windowed_percentile > 0
    interval_sum = 0.0
    interval_count = 0
    avgs: list[float] = []
    # Rolling window of raw per-epoch losses for percentile mode.
    raw_window: list[float] = []
    epochs_since_check = 0
    conv_count = 0

    for ep in range(config.total_epochs):
        config.before_epoch(ep)
        loss = epoch_fn()
        if config.lr_scheduler is not None:
            config.lr_scheduler.step()
        epochs_done = ep + 1

        # Log progress
        if config.log_every > 0 and ep % config.log_every == 0:
            extra = metrics_fn() if metrics_fn else []
            print(
                f"  {format_elapsed(t_start)} {ep}\tloss={loss:.6f}"
                f"{mem_suffix()}{_format_metrics(extra)}"
            )

        # NaN detection
        if math.isnan(loss):
            print(f"  {format_elapsed(t_start)} Diverged (NaN) at epoch {ep}")
            return epochs_done, loss

        final_loss = loss

        # Patience-based early stopping
        if config.patience > 0 and not use_windowed:
            if loss < best_loss - config.min_delta:
                best_loss = loss
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                print(
                    f"  {format_elapsed(t_start)} Early stop at epoch {epochs_done}"
                    f" (patience={config.patience})"
                )
                break

        # Windowed-average early stopping (mean over 100-epoch chunks)
        if use_windowed and not use_percentile:
            interval_sum += loss
            interval_count += 1
            if interval_count >= 100:
                avg = interval_sum / 100.0
                avgs.append(avg)
                interval_sum = 0.0
                interval_count = 0
                wc = max(1, config.windowed_window // 100)
                if len(avgs) >= wc:
                    window_avg = sum(avgs[-wc:]) / wc
                    if window_avg < config.windowed_threshold:
                        conv_count += 1
                        if conv_count >= config.windowed_patience:
                            print(
                                f"  {format_elapsed(t_start)} Converged at epoch {epochs_done}"
                                f" (window_avg={window_avg})"
                            )
                            break
                        print(
                            f"    {format_elapsed(t_start)} convergence"
                            f" {conv_count}/{config.windowed_patience}"
                            f" (window_avg={window_avg})"
                        )
                    else:
                        conv_count = 0

        # Windowed-percentile early stopping (kth percentile of raw losses).
        # Mirrors Idris `goWindowedPercentile` exactly: keep the last `win`
        # raw losses, sort every 100 epochs once the window is full, pick
        # idx = min(win-1, floor(pct * win)), compare to threshold.
        if use_percentile:
            raw_window.append(loss)
            if len(raw_window) > config.windowed_window:
                raw_window = raw_window[-config.windowed_window:]
            epochs_since_check += 1
            if epochs_since_check >= 100 and len(raw_window) >= config.windowed_window:
                epochs_since_check = 0
                sorted_w = sorted(raw_window)
                idx = min(
                    config.windowed_window - 1,
                    int(config.windowed_percentile * config.windowed_window),
                )
                pct_val = sorted_w[idx]
                pct_label = f"p{int(config.windowed_percentile * 100)}_loss"
                if pct_val < config.windowed_threshold:
                    conv_count += 1
                    if conv_count >= config.windowed_patience:
                        print(
                            f"  {format_elapsed(t_start)} Converged at epoch {epochs_done}"
                            f" ({pct_label}={pct_val})"
                        )
                        break
                    print(
                        f"    {format_elapsed(t_start)} convergence"
                        f" {conv_count}/{config.windowed_patience}"
                        f" ({pct_label}={pct_val})"
                    )
                else:
                    conv_count = 0

    total_elapsed = time.monotonic() - t_start
    total_sec = int(total_elapsed)
    dur = _format_duration(total_sec)
    ms_per = (total_sec * 1000) // epochs_done if epochs_done > 0 else 0
    print(f"Completed in {dur} ({epochs_done} epochs, {ms_per}ms/epoch)")
    # In-script timing marker for perf-baseline.sh — float ms/epoch over
    # the timed training loop only, no startup / eval / build overhead.
    ms_per_ep_float = total_elapsed * 1000.0 / epochs_done if epochs_done > 0 else 0.0
    print(f"PERF_MS_PER_EP={ms_per_ep_float:.6f}")

    return epochs_done, final_loss

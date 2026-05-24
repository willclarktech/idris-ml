"""Unified training runner matching Idris runTraining output format.

Handles epoch loop, progress logging, early stopping, timing, and
result formatting. Output is identical to the Idris Train.idr runner.
"""

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

_PROC = psutil.Process()
_PEAK_MB = 0

# Module-level device + dtype singletons. Model/loop code calls
# `get_device()` / `get_dtype()` to place new tensors on the active
# device/dtype without threading the values through every signature.
# `run_training` calls `set_device(config.device)` before the first
# epoch; each script's main() can also call it directly when
# constructing tensors before the training loop starts.
#
# Dtype is auto-selected from device: torch.float64 for cpu/cuda,
# torch.float32 for mps (libtorch's MPS backend rejects float64 at
# tensor construction with `Cannot convert a MPS Tensor to float64
# dtype`, so we silently downcast). This mirrors idris-ml's
# `(MlxDev MGpu) F32` only / `MlxDev MCpu` F32+F64 design.
_DEVICE: str = "cpu"
_DTYPE: torch.dtype = torch.float64


def _dtype_for_device(d: str) -> torch.dtype:
    """Default dtype per device: F32 on MPS (libtorch rejects F64),
    F64 elsewhere (refs' historical default for numerical parity with
    idris-ml's F64 default)."""
    return torch.float32 if d == "mps" else torch.float64


def set_device(d: str) -> None:
    """Set the active device and auto-derive the dtype."""
    global _DEVICE, _DTYPE
    _DEVICE = d
    _DTYPE = _dtype_for_device(d)


def get_device() -> str:
    """Active device string ("cpu" / "mps" / "cuda") for tensor creation."""
    return _DEVICE


def get_dtype() -> torch.dtype:
    """Active floating-point dtype, auto-derived from `get_device()`."""
    return _DTYPE


def multinomial_safe(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """`torch.multinomial` with explicit MPS workaround.

    `torch.multinomial` has no MPS kernel — on MPS tensors PyTorch
    silently falls back to CPU under `PYTORCH_ENABLE_MPS_FALLBACK=1`
    (the recent default). Making the round-trip explicit here keeps
    the cost visible in profiles and grep-able in code.
    """
    if probs.device.type == "mps":
        return torch.multinomial(probs.cpu(), num_samples).to("mps")
    return torch.multinomial(probs, num_samples)


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
    # Device for tensor placement: "cpu" / "mps" / "cuda". Threaded into
    # the module-level `_DEVICE` singleton at the start of run_training so
    # `get_device()` calls inside model/loop code see the right value.
    device: str = "cpu"
    # Optional checkpoint policy (mirrors Idris TrainConfig.checkpoint).
    # When set, run_training resumes from `<dir>/last` before the loop,
    # periodically saves + keeps the best checkpoint after each non-NaN
    # epoch, and reloads the best at the end (return-best semantics).
    checkpoint: "CheckpointPolicy | None" = None


@dataclass
class CheckpointPolicy:
    """File-backed checkpoint policy, parallel to Idris `CheckpointPolicy`.

    Build with `file_checkpoint`. `save_state(prefix, epoch, best)` writes
    the model + optimizer + a `trainer_state.json` sidecar under
    `<prefix>.*`; `load_state(prefix)` restores them and returns
    `(resume_epoch, best_metric)` or None for a fresh start. `monitor`
    selects the keep-best scalar (lower is better); None tracks the
    per-epoch training loss.
    """

    dir: str
    every_n: int
    keep_best: bool
    save_state: Callable[[str, int, float], None]
    load_state: Callable[[str], "tuple[int, float] | None"]
    monitor: Callable[[], float] | None = None


def file_checkpoint(
    directory: str,
    every_n: int,
    keep_best: bool,
    model: Any,
    optimizer: Any,
    monitor: Callable[[], float] | None = None,
) -> CheckpointPolicy:
    """Build a file-backed checkpoint policy closing over model + optimizer.

    Mirrors Idris `fileCheckpoint`. Periodic saves use the `<dir>/last`
    prefix; keep-best uses `<dir>/best`. Resume metadata rides in an
    HF-Trainer-style `trainer_state.json` sidecar.
    """
    os.makedirs(directory, exist_ok=True)

    def save_state(prefix: str, epoch: int, best: float) -> None:
        torch.save(model.state_dict(), prefix + ".model.pt")
        torch.save(optimizer.state_dict(), prefix + ".opt.pt")
        with open(prefix + ".trainer_state.json", "w") as f:
            json.dump({"epoch": epoch, "best": best, "timestamp": int(time.time())}, f)

    def load_state(prefix: str) -> "tuple[int, float] | None":
        sidecar = prefix + ".trainer_state.json"
        if not os.path.exists(sidecar):
            return None
        with open(sidecar) as f:
            st = json.load(f)
        model.load_state_dict(torch.load(prefix + ".model.pt"))
        optimizer.load_state_dict(torch.load(prefix + ".opt.pt"))
        return int(st["epoch"]), float(st["best"])

    return CheckpointPolicy(directory, every_n, keep_best, save_state, load_state, monitor)


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
    set_device(config.device)
    t_start = time.monotonic()
    print("Training...")

    best_loss = math.inf
    stale = 0
    final_loss = 0.0
    epochs_done = 0

    # Checkpoint resume: load <dir>/last before the loop, seed start epoch
    # + best metric. `ckpt_best` is the keep-best watermark (mirrors the
    # Idris `bestRef` IORef, separate from patience's `best_loss`).
    ckpt = config.checkpoint
    start_ep = 0
    ckpt_best = math.inf
    if ckpt is not None:
        st = ckpt.load_state(os.path.join(ckpt.dir, "last"))
        if st is not None:
            start_ep, ckpt_best = st
            print(f"  Resuming from epoch {start_ep} (best={ckpt_best:.6f})")

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

    for ep in range(start_ep, config.total_epochs):
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

        # Checkpoint after each non-NaN epoch: keep-best to <dir>/best,
        # then periodic to <dir>/last. The sidecar stores `ep + 1` as the
        # resume point. Mirrors Idris `postEpoch`.
        if ckpt is not None:
            if ckpt.keep_best:
                cur = ckpt.monitor() if ckpt.monitor is not None else loss
                if cur < ckpt_best:
                    ckpt_best = cur
                    ckpt.save_state(os.path.join(ckpt.dir, "best"), ep + 1, cur)
            if ckpt.every_n > 0 and (ep + 1) % ckpt.every_n == 0:
                ckpt.save_state(os.path.join(ckpt.dir, "last"), ep + 1, ckpt_best)

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
                raw_window = raw_window[-config.windowed_window :]
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

    # Return-best: reload the best checkpoint so the post-training model
    # is the best seen, not the last (Lightning semantics, mirrors Idris).
    if ckpt is not None and ckpt.keep_best:
        ckpt.load_state(os.path.join(ckpt.dir, "best"))

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

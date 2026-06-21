#!/usr/bin/env python3
"""Contract tests for the perf-regression gate (`scripts/check-perf-regression.py`).

Locks the gating behaviour the perf.yml `bench-deep` job depends on:

  - `run` mode is advisory unless `PERF_GATE=1` — a FAIL exits 0 without the
    env, 1 with it. This is the switch perf.yml flips to enforce.
  - A cell with fewer than N+1 samples is INSUFFICIENT-HISTORY and can never
    FAIL, even under `PERF_GATE=1`. This is what makes early enforcement safe
    while per-cell history is still accruing.
  - `op_bench` mode always gates (`always_gates=True`) — a FAIL exits 1
    regardless of `PERF_GATE`.

Drives the script as a subprocess so the real `PERF_GATE` env + exit-code
path is exercised end-to-end, over a synthetic temp JSONL.

Run via:
    python3 scripts/tests/test_check_perf_regression.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check-perf-regression.py"

# Window sizes the script medians over (see check-perf-regression.py).
RUN_WINDOW = 10
OP_BENCH_WINDOW = 5


def _run_entry(wall_ms: int, *, example: str = "ex", backend: str = "tape") -> dict[str, object]:
    return {
        "kind": "run",
        "example": example,
        "backend": backend,
        "args": "",
        "wall_ms": wall_ms,
        "exit": 0,
        "commit": "test",
    }


def _op_bench_entry(ms_per_iter: float, *, label: str = "wl") -> dict[str, object]:
    return {
        "kind": "op_bench",
        "axis": "A",
        "label": label,
        "runtime": "tape",
        "ms_per_iter": ms_per_iter,
        "exit": 0,
        "commit": "test",
    }


def _write_log(entries: list[dict[str, object]]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return Path(path)


def _invoke(log: Path, mode: str, *, perf_gate: bool) -> int:
    env = dict(os.environ)
    if perf_gate:
        env["PERF_GATE"] = "1"
    else:
        env.pop("PERF_GATE", None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", mode, "--log", str(log)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode


def test_run_fail_is_advisory_without_perf_gate() -> None:
    # 10 baseline entries at 100ms, then a 300ms current (+200% > 100% FAIL).
    entries = [_run_entry(100) for _ in range(RUN_WINDOW)] + [_run_entry(300)]
    log = _write_log(entries)
    try:
        assert _invoke(log, "run", perf_gate=False) == 0, (
            "run FAIL must be advisory (exit 0) without PERF_GATE"
        )
    finally:
        log.unlink()


def test_run_fail_gates_under_perf_gate() -> None:
    entries = [_run_entry(100) for _ in range(RUN_WINDOW)] + [_run_entry(300)]
    log = _write_log(entries)
    try:
        assert _invoke(log, "run", perf_gate=True) == 1, "run FAIL must exit 1 under PERF_GATE=1"
    finally:
        log.unlink()


def test_insufficient_history_never_fails() -> None:
    # Only 3 samples (< RUN_WINDOW+1): a huge jump must NOT fail — the cell is
    # INSUFFICIENT-HISTORY. This is what makes enabling PERF_GATE=1 safe today.
    entries = [_run_entry(100), _run_entry(100), _run_entry(100000)]
    log = _write_log(entries)
    try:
        assert _invoke(log, "run", perf_gate=True) == 0, "insufficient-history cell must never FAIL"
    finally:
        log.unlink()


def test_run_ok_passes_under_perf_gate() -> None:
    # 10 baseline at 100ms, current 105ms (+5%, within the ±15% OK band).
    entries = [_run_entry(100) for _ in range(RUN_WINDOW)] + [_run_entry(105)]
    log = _write_log(entries)
    try:
        assert _invoke(log, "run", perf_gate=True) == 0, "run OK must pass even under PERF_GATE=1"
    finally:
        log.unlink()


def test_op_bench_always_gates() -> None:
    # 5 baseline at 1.0 ms/iter, current 2.0 (+100% > 40% FAIL). op_bench
    # gates regardless of PERF_GATE (always_gates=True).
    entries = [_op_bench_entry(1.0) for _ in range(OP_BENCH_WINDOW)] + [_op_bench_entry(2.0)]
    log = _write_log(entries)
    try:
        assert _invoke(log, "op_bench", perf_gate=False) == 1, (
            "op_bench FAIL must exit 1 without PERF_GATE"
        )
        assert _invoke(log, "op_bench", perf_gate=True) == 1, (
            "op_bench FAIL must exit 1 with PERF_GATE"
        )
    finally:
        log.unlink()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")

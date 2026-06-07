#!/usr/bin/env python3
"""Perf-regression CI gate over docs/develop/perf-log.jsonl.

Two modes, picked by `--mode`:

  `op_bench` (default): kernel-microbench entries. Groups by
    `(axis, label, runtime)`, metric = `ms_per_iter`. Hard-fails
    on regressions.

  `run`: end-to-end example-run entries. Groups by
    `(example, backend, args)`, metric = `wall_ms`. Gated behind
    `PERF_GATE=1` env var (informational only) until the noise
    profile is calibrated; promote to hard-fail by setting the env.

For each cell, treats the latest entry as the current measurement
and computes the median of the preceding N entries as the baseline
(N=5 for op_bench, N=10 for run). Classifies the current value:

    OK    — within ±15% of baseline (the documented VM noise floor;
            see feedback_vm_perf_noise.md).
    WARN  — > 15% slower, ≤ R% (R = FAIL_PCT, mode-specific).
    FAIL  — > R% slower than baseline.

Cells with fewer than N+1 samples are reported as INSUFFICIENT-HISTORY.
Faster-than-baseline current entries are always OK (no warn/fail for
performance improvements).

Prints a Markdown verdict table to stdout. Exits 1 if any cell is
FAIL AND the mode is gating (op_bench always gates; run gates only
when PERF_GATE=1). Thresholds were calibrated against three
perf-changes.md noise-floor entries from 2026-06-03 and the
`feedback_vm_perf_noise.md` memory.

Usage:
    python3 scripts/check-perf-regression.py
    python3 scripts/check-perf-regression.py --mode run
    PERF_GATE=1 python3 scripts/check-perf-regression.py --mode run
    python3 scripts/check-perf-regression.py --baseline-window 7
    python3 scripts/check-perf-regression.py --log path/to/perf-log.jsonl

Schema (reads):
    op_bench mode:
      {"kind": "op_bench", "axis": "A"|"B"|"C"|"D",
       "label": "...", "runtime": "tape"|"pytorch",
       "ms_per_iter": <float>, "commit": "...", "ts": "...", ...}
    run mode:
      {"kind": "run", "example": "...", "backend": "tape"|"torch"|"mlx",
       "args": "...", "wall_ms": <int>, "commit": "...", "ts": "...", ...}
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from mltools.perf_log import iter_entries, resolve_log_path  # noqa: E402

DEFAULT_LOG = resolve_log_path()

# Thresholds: percent slower than baseline. Calibrated against VM noise
# (±15-20%) per docs/develop/perf-changes.md 2026-06-03 entries and
# `feedback_vm_perf_noise.md`.
WARN_PCT = 15.0
OP_BENCH_FAIL_PCT = 40.0
# Run-mode threshold is wider (100% = 2× slower) because example-run
# wall_ms carries more variance than per-op microbench ms_per_iter:
# build cache state, multi-iteration averaging, varied workload size.
# Tighten once the noise profile is calibrated.
RUN_FAIL_PCT = 100.0

# How many prior entries to median over for the baseline.
OP_BENCH_BASELINE_WINDOW = 5
RUN_BASELINE_WINDOW = 10


# Per-mode wiring: how to filter, group, and extract the metric.
def _key_op_bench(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (entry.get("axis", ""), entry.get("label", ""), entry.get("runtime", ""))


def _key_run(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (entry.get("example", ""), entry.get("backend", ""), entry.get("args", ""))


class ModeConfig(TypedDict):
    metric: str
    metric_label: str
    key_fn: Callable[[dict[str, Any]], tuple[str, str, str]]
    key_labels: tuple[str, str, str]
    fail_pct: float
    baseline_window: int
    always_gates: bool


MODES: dict[str, ModeConfig] = {
    "op_bench": {
        "metric": "ms_per_iter",
        "metric_label": "ms/iter",
        "key_fn": _key_op_bench,
        "key_labels": ("Axis", "Workload", "Runtime"),
        "fail_pct": OP_BENCH_FAIL_PCT,
        "baseline_window": OP_BENCH_BASELINE_WINDOW,
        "always_gates": True,
    },
    "run": {
        "metric": "wall_ms",
        "metric_label": "wall_ms",
        "key_fn": _key_run,
        "key_labels": ("Example", "Backend", "Args"),
        "fail_pct": RUN_FAIL_PCT,
        "baseline_window": RUN_BASELINE_WINDOW,
        "always_gates": False,  # respect PERF_GATE env var
    },
}


def load_entries(log_path: Path, kind: str, metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in iter_entries(log_path):
        if entry.get("kind") != kind:
            continue
        if entry.get(metric) is None:
            continue
        # Skip failed runs (exit != 0) — they don't represent steady-state
        # perf and would skew the baseline.
        if entry.get("exit") not in (None, 0):
            continue
        out.append(entry)
    return out


def group_by_cell(
    entries: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[str, str, str]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Group entries by the per-mode key function. Preserves the JSONL
    order within each cell (append-only log → naturally time-sorted)."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        cells[key_fn(entry)].append(entry)
    return cells


def classify(current: float, baseline: float, fail_pct: float) -> tuple[str, float]:
    """Return (verdict, delta_pct). delta_pct is positive when current
    is slower than baseline."""
    if baseline <= 0.0:
        return ("INDETERMINATE", 0.0)
    delta_pct = (current - baseline) / baseline * 100.0
    if delta_pct <= WARN_PCT:
        return ("OK", delta_pct)
    if delta_pct <= fail_pct:
        return ("WARN", delta_pct)
    return ("FAIL", delta_pct)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="op_bench",
        help="Which kind of entries to gate over (default: op_bench).",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Path to perf-log.jsonl (default: {DEFAULT_LOG})",
    )
    p.add_argument(
        "--baseline-window",
        type=int,
        default=None,
        help="How many prior entries to median for baseline "
        "(mode-specific default: 5 for op_bench, 10 for run).",
    )
    args = p.parse_args()

    mode_cfg = MODES[args.mode]
    metric = mode_cfg["metric"]
    baseline_n = args.baseline_window or mode_cfg["baseline_window"]
    fail_pct = mode_cfg["fail_pct"]

    entries = load_entries(args.log, kind=args.mode, metric=metric)
    if not entries:
        print(f"# Perf regression gate ({args.mode})")
        print()
        print(f'_No `kind: "{args.mode}"` entries in {args.log} yet._')
        return 0

    cells = group_by_cell(entries, mode_cfg["key_fn"])

    rows: list[dict[str, Any]] = []
    counts = {"OK": 0, "WARN": 0, "FAIL": 0, "INSUFFICIENT": 0, "INDETERMINATE": 0}
    for key in sorted(cells.keys()):
        cell_entries = cells[key]
        if len(cell_entries) < baseline_n + 1:
            verdict = "INSUFFICIENT"
            current = cell_entries[-1][metric]
            baseline_str = f"n={len(cell_entries)}"
            delta_str = "—"
            counts["INSUFFICIENT"] += 1
        else:
            window = cell_entries[-(baseline_n + 1) : -1]
            current_entry = cell_entries[-1]
            current = current_entry[metric]
            baseline = statistics.median(e[metric] for e in window)
            verdict, delta_pct = classify(current, baseline, fail_pct)
            baseline_str = f"{baseline:.4f}" if args.mode == "op_bench" else f"{baseline:.0f}"
            sign = "+" if delta_pct >= 0 else ""
            delta_str = f"{sign}{delta_pct:.1f}%"
            counts[verdict] = counts.get(verdict, 0) + 1
        rows.append(
            {
                "k0": key[0],
                "k1": key[1],
                "k2": key[2],
                "current": current,
                "baseline": baseline_str,
                "delta": delta_str,
                "verdict": verdict,
                "commit": cell_entries[-1].get("commit", "?"),
            }
        )

    # Markdown verdict table.
    print(f"# Perf regression gate ({args.mode})")
    print()
    print(f"Source: `{args.log.relative_to(ROOT) if args.log.is_relative_to(ROOT) else args.log}`")
    print(f"Baseline window: median of prior {baseline_n} entries per cell.")
    print(f"Thresholds: ±{WARN_PCT:g}% (OK), > {WARN_PCT:g}% (WARN), > {fail_pct:g}% (FAIL).")
    print()
    print(
        f"Counts: OK={counts['OK']}, WARN={counts['WARN']}, "
        f"FAIL={counts['FAIL']}, "
        f"INSUFFICIENT-HISTORY={counts['INSUFFICIENT']}."
    )
    print()
    k0_label, k1_label, k2_label = mode_cfg["key_labels"]
    metric_col = (f"Baseline ({mode_cfg['metric_label']})", f"Current ({mode_cfg['metric_label']})")
    cur_fmt = "{:.4f}" if args.mode == "op_bench" else "{:.0f}"
    print(
        f"| {k0_label} | {k1_label} | {k2_label} | {metric_col[0]} "
        f"| {metric_col[1]} | Delta | Verdict | Commit |"
    )
    print("|---|---|---|---:|---:|---:|---|---|")
    for r in rows:
        print(
            f"| {r['k0']} | {r['k1']} | {r['k2']} "
            f"| {r['baseline']} | {cur_fmt.format(r['current'])} | {r['delta']} "
            f"| {r['verdict']} | `{r['commit']}` |"
        )
    print()

    # In gating modes (op_bench always; run only when PERF_GATE=1) a
    # FAIL means exit 1. Otherwise the report is informational.
    if counts["FAIL"] > 0 and (mode_cfg["always_gates"] or os.environ.get("PERF_GATE") == "1"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Perf-regression CI gate over docs/develop/perf-log.jsonl.

Loads every `kind: "op_bench"` entry, groups by `(axis, label, runtime)`,
treats the latest entry as the current measurement, and computes the
median of the preceding N entries as the baseline (N=5 by default).
For each cell classifies the current ms/iter as:

    OK    — within ±15% of baseline (the documented VM noise floor;
            see feedback_vm_perf_noise.md).
    WARN  — > 15% slower, ≤ 40%.
    FAIL  — > 40% slower than baseline.

Cells with fewer than N+1 samples are reported as INSUFFICIENT-HISTORY.
Faster-than-baseline current entries are always OK (no warn/fail for
performance improvements).

Prints a Markdown verdict table to stdout.

Phase 5a (this commit) — exit code is always 0; the table is advisory.
Phase 5b will promote the WARN threshold to a printed warning while
keeping exit 0, and the FAIL threshold to exit 1 (CI red). Thresholds
were calibrated against three perf-changes.md noise-floor entries from
2026-06-03 and the existing `feedback_vm_perf_noise.md` memory.

Usage:
    python3 scripts/check-perf-regression.py
    python3 scripts/check-perf-regression.py --baseline-window 7
    python3 scripts/check-perf-regression.py --log path/to/perf-log.jsonl

Schema (reads):
    {"kind": "op_bench", "axis": "A"|"B"|"C"|"D",
     "label": "...", "runtime": "tape"|"pytorch",
     "ms_per_iter": <float>, "commit": "...", "ts": "...", ...}
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "docs" / "develop" / "perf-log.jsonl"

# Thresholds: percent slower than baseline. Calibrated against VM noise
# (±15-20%) per docs/develop/perf-changes.md 2026-06-03 entries and
# `feedback_vm_perf_noise.md`.
WARN_PCT = 15.0
FAIL_PCT = 40.0

# How many prior entries to median over for the baseline.
DEFAULT_BASELINE_WINDOW = 5


def load_op_bench_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("kind") != "op_bench":
            continue
        if entry.get("ms_per_iter") is None:
            continue
        out.append(entry)
    return out


def group_by_cell(entries: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Group entries by (axis, label, runtime). Preserves the JSONL
    order within each cell (append-only log → naturally time-sorted)."""
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        key = (entry.get("axis", ""), entry.get("label", ""),
               entry.get("runtime", ""))
        cells[key].append(entry)
    return cells


def classify(current: float, baseline: float) -> tuple[str, float]:
    """Return (verdict, delta_pct). delta_pct is positive when current
    is slower than baseline."""
    if baseline <= 0.0:
        return ("INDETERMINATE", 0.0)
    delta_pct = (current - baseline) / baseline * 100.0
    if delta_pct <= WARN_PCT:
        return ("OK", delta_pct)
    if delta_pct <= FAIL_PCT:
        return ("WARN", delta_pct)
    return ("FAIL", delta_pct)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG,
                   help=f"Path to perf-log.jsonl (default: {DEFAULT_LOG})")
    p.add_argument("--baseline-window", type=int,
                   default=DEFAULT_BASELINE_WINDOW,
                   help=f"How many prior entries to median for baseline "
                        f"(default: {DEFAULT_BASELINE_WINDOW})")
    args = p.parse_args()

    entries = load_op_bench_entries(args.log)
    if not entries:
        print(f"# Perf regression gate")
        print()
        print(f"_No `kind: \"op_bench\"` entries in {args.log} yet._")
        return 0

    cells = group_by_cell(entries)
    N = args.baseline_window

    rows = []
    counts = {"OK": 0, "WARN": 0, "FAIL": 0,
              "INSUFFICIENT": 0, "INDETERMINATE": 0}
    for key in sorted(cells.keys()):
        axis, label, runtime = key
        cell_entries = cells[key]
        if len(cell_entries) < N + 1:
            verdict = "INSUFFICIENT"
            current = cell_entries[-1]["ms_per_iter"]
            baseline_str = f"n={len(cell_entries)}"
            delta_str = "—"
            counts["INSUFFICIENT"] += 1
        else:
            window = cell_entries[-(N + 1):-1]
            current_entry = cell_entries[-1]
            current = current_entry["ms_per_iter"]
            baseline = statistics.median(
                e["ms_per_iter"] for e in window)
            verdict, delta_pct = classify(current, baseline)
            baseline_str = f"{baseline:.4f}"
            sign = "+" if delta_pct >= 0 else ""
            delta_str = f"{sign}{delta_pct:.1f}%"
            counts[verdict] = counts.get(verdict, 0) + 1
        rows.append({
            "axis": axis,
            "label": label,
            "runtime": runtime,
            "current": current,
            "baseline": baseline_str,
            "delta": delta_str,
            "verdict": verdict,
            "commit": cell_entries[-1].get("commit", "?"),
        })

    # Markdown verdict table.
    print(f"# Perf regression gate")
    print()
    print(f"Source: `{args.log.relative_to(ROOT) if args.log.is_relative_to(ROOT) else args.log}`")
    print(f"Baseline window: median of prior {N} entries per cell.")
    print(f"Thresholds: ±{WARN_PCT:g}% (OK), > {WARN_PCT:g}% (WARN), > {FAIL_PCT:g}% (FAIL).")
    print()
    print(f"Counts: OK={counts['OK']}, WARN={counts['WARN']}, "
          f"FAIL={counts['FAIL']}, "
          f"INSUFFICIENT-HISTORY={counts['INSUFFICIENT']}.")
    print()
    print("| Axis | Workload | Runtime | Baseline (ms/iter) | Current (ms/iter) | Delta | Verdict | Commit |")
    print("|---|---|---|---:|---:|---:|---|---|")
    for r in rows:
        print(f"| {r['axis']} | {r['label']} | {r['runtime']} "
              f"| {r['baseline']} | {r['current']:.4f} | {r['delta']} "
              f"| {r['verdict']} | `{r['commit']}` |")
    print()

    # Phase 5a: advisory only. Exit 0 always so the gate runs as a
    # warm-up in CI without blocking merges. Phase 5b (a separate
    # commit, ~2 weeks later) flips FAIL to exit 1.
    return 0


if __name__ == "__main__":
    sys.exit(main())

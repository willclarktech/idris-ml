#!/usr/bin/env bash
# scripts/perf-fast.sh — Tier 1 perf gate (Axis A op-kernel + Axis B
# single-layer fwd+bwd, tape only).
#
# Runs four benchmark commands in sequence:
#   - make bench-ops      (Axis A, C-kernel level)
#   - make bench-ops-py   (Axis A, PyTorch ref)
#   - make bench-layers   (Axis B, Idris-side layer fwd+bwd+step)
#   - make bench-layers-py (Axis B, PyTorch ref)
# Parses each "<label>:\t<ms> ms (<iters> iters)" line into a structured
# JSON entry, appends one entry per measurement to
# docs/develop/perf-log.jsonl with kind="op_bench" and axis ∈ {A, B}.
# The label format is identical across both axes; the axis assignment
# comes from which input file the entry was parsed from.
#
# Then regenerates BENCHMARKS.md so the repo-front-page comparison
# table reflects the latest commit. Designed to run in ≤ 5 min on
# CI so it can gate every PR.
#
# Schema (new entries):
#   { "kind": "op_bench", "axis": "A" | "B", "section": "...",
#     "label": "...", "runtime": "tape" | "pytorch",
#     "wall_ms": ..., "iters": ..., "ms_per_iter": ...,
#     "commit": "...", "ts": "...", "date": "..." }
#
# Usage:
#   scripts/perf-fast.sh            # build + run + log + render
#   scripts/perf-fast.sh --no-build # assume bench_ops is already built
#   scripts/perf-fast.sh --no-render # skip BENCHMARKS.md regeneration

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_PATH="docs/develop/perf-log.jsonl"
BENCHMARKS_PATH="BENCHMARKS.md"

DO_BUILD=1
DO_RENDER=1
for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    --no-render) DO_RENDER=0 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "perf-fast.sh: unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [ -n "$(git status --porcelain -- ':!docs/develop/perf-log.jsonl' ':!BENCHMARKS.md' 2>/dev/null)" ]; then
  COMMIT="${COMMIT}+dirty"
fi
ISO_TS=$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')
DATE=$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%d"))')

A_TAPE_OUT=$(mktemp)
A_PYTORCH_OUT=$(mktemp)
B_TAPE_OUT=$(mktemp)
B_PYTORCH_OUT=$(mktemp)
trap 'rm -f "$A_TAPE_OUT" "$A_PYTORCH_OUT" "$B_TAPE_OUT" "$B_PYTORCH_OUT"' EXIT

echo "==> perf-fast: running tape Axis A (op kernels)"
if [ "$DO_BUILD" = "1" ]; then
  caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make bench-ops 2>&1 | tee "$A_TAPE_OUT" >&2
else
  ./build/tape-mlxcpu-torchcpu/bench_ops | tee "$A_TAPE_OUT" >&2
fi

echo "==> perf-fast: running pytorch Axis A (op kernels)"
caffeinate -i nice -n 19 make bench-ops-py 2>&1 | tee "$A_PYTORCH_OUT" >&2

echo "==> perf-fast: running tape Axis B (single-layer fwd+bwd)"
if [ "$DO_BUILD" = "1" ]; then
  caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make bench-layers 2>&1 | tee "$B_TAPE_OUT" >&2
else
  ./build/tape-mlxcpu-torchcpu/exec/layers-bench | tee "$B_TAPE_OUT" >&2
fi

echo "==> perf-fast: running pytorch Axis B (single-layer fwd+bwd)"
caffeinate -i nice -n 19 make bench-layers-py 2>&1 | tee "$B_PYTORCH_OUT" >&2

echo "==> perf-fast: parsing and logging measurements"
COMMIT="$COMMIT" ISO_TS="$ISO_TS" DATE="$DATE" \
A_TAPE_OUT="$A_TAPE_OUT" A_PYTORCH_OUT="$A_PYTORCH_OUT" \
B_TAPE_OUT="$B_TAPE_OUT" B_PYTORCH_OUT="$B_PYTORCH_OUT" \
LOG_PATH="$LOG_PATH" \
python3 <<'PY'
import json, os, re

LINE_RE = re.compile(r"^([A-Za-z][^\t:]*?):\s*([0-9.]+)\s*ms\s*\((\d+)\s*iters\)\s*$")
SECTION_RE = re.compile(r"^---\s*(.+?)\s*---\s*$")

def parse_runtime_output(path, runtime, axis, *, commit, iso_ts, date):
    entries = []
    section = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = SECTION_RE.match(line)
            if m:
                section = m.group(1)
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            label, wall_ms, iters = m.group(1).strip(), float(m.group(2)), int(m.group(3))
            entries.append({
                "ts": iso_ts,
                "date": date,
                "kind": "op_bench",
                "axis": axis,
                "section": section or "",
                "label": label,
                "runtime": runtime,
                "commit": commit,
                "wall_ms": wall_ms,
                "iters": iters,
                "ms_per_iter": wall_ms / iters if iters else None,
            })
    return entries

commit = os.environ["COMMIT"]
iso_ts = os.environ["ISO_TS"]
date = os.environ["DATE"]
all_entries = []
for path_var, runtime, axis in [
    ("A_TAPE_OUT",   "tape",    "A"),
    ("A_PYTORCH_OUT","pytorch", "A"),
    ("B_TAPE_OUT",   "tape",    "B"),
    ("B_PYTORCH_OUT","pytorch", "B"),
]:
    all_entries.extend(
        parse_runtime_output(os.environ[path_var], runtime, axis,
                             commit=commit, iso_ts=iso_ts, date=date)
    )
log_path = os.environ["LOG_PATH"]
with open(log_path, "a") as out:
    for entry in all_entries:
        out.write(json.dumps(entry) + "\n")
by_axis = {}
for e in all_entries:
    by_axis.setdefault(e["axis"], 0)
    by_axis[e["axis"]] += 1
summary = ", ".join(f"{n} axis-{a}" for a, n in sorted(by_axis.items()))
print(f"appended {len(all_entries)} entries ({summary}) to {log_path}")
PY

if [ "$DO_RENDER" = "1" ]; then
  echo "==> perf-fast: rendering BENCHMARKS.md"
  python3 scripts/render-benchmarks.py
fi

echo "==> perf-fast: done"

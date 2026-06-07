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
# Parsing + append delegate to `mltools.perf_log parse-op-bench`.
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

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"

cd "$PERF_REPO_ROOT"

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

COMMIT=$( perf_commit_with_dirty )

A_TAPE_OUT=$( mktemp )
A_PYTORCH_OUT=$( mktemp )
B_TAPE_OUT=$( mktemp )
B_PYTORCH_OUT=$( mktemp )
trap 'rm -f "$A_TAPE_OUT" "$A_PYTORCH_OUT" "$B_TAPE_OUT" "$B_PYTORCH_OUT"' EXIT

echo "==> perf-fast: running tape Axis A (op kernels)"
if [ "$DO_BUILD" = "1" ]; then
	perf_quiet_run env MAKEFLAGS=-j2 make bench-ops 2>&1 | tee "$A_TAPE_OUT" >&2
else
	./build/tape-mlxcpu-torchcpu/bench_ops | tee "$A_TAPE_OUT" >&2
fi

echo "==> perf-fast: running pytorch Axis A (op kernels)"
perf_quiet_run make bench-ops-py 2>&1 | tee "$A_PYTORCH_OUT" >&2

echo "==> perf-fast: running tape Axis B (single-layer fwd+bwd)"
if [ "$DO_BUILD" = "1" ]; then
	perf_quiet_run env MAKEFLAGS=-j2 make bench-layers 2>&1 | tee "$B_TAPE_OUT" >&2
else
	./build/tape-mlxcpu-torchcpu/exec/layers-bench | tee "$B_TAPE_OUT" >&2
fi

echo "==> perf-fast: running pytorch Axis B (single-layer fwd+bwd)"
perf_quiet_run make bench-layers-py 2>&1 | tee "$B_PYTORCH_OUT" >&2

echo "==> perf-fast: parsing and logging measurements"
python3 -m mltools.perf_log parse-op-bench --axis A --runtime tape \
	--commit "$COMMIT" --input "$A_TAPE_OUT"
python3 -m mltools.perf_log parse-op-bench --axis A --runtime pytorch \
	--commit "$COMMIT" --input "$A_PYTORCH_OUT"
python3 -m mltools.perf_log parse-op-bench --axis B --runtime tape \
	--commit "$COMMIT" --input "$B_TAPE_OUT"
python3 -m mltools.perf_log parse-op-bench --axis B --runtime pytorch \
	--commit "$COMMIT" --input "$B_PYTORCH_OUT"

if [ "$DO_RENDER" = "1" ]; then
	echo "==> perf-fast: rendering BENCHMARKS.md"
	python3 scripts/render-benchmarks.py
fi

echo "==> perf-fast: done"

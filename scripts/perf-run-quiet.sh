#!/usr/bin/env bash
# scripts/perf-run-quiet.sh — wrapper around perf-run.sh that's gentler
# on the host while a build runs. Use when you want to keep using the
# machine for other work and don't mind the build taking a bit longer.
#
# Applies, in order:
#   1. `caffeinate -i` (macOS only) — prevents idle sleep while the
#      build runs. Necessary on laptops where a closed lid or display
#      sleep can suspend/kill long builds mid-run.
#   2. `nice -n 19` — lowest CPU priority (range is -20..+20; +19 is
#      "give me CPU only when nothing else wants it"). Foreground apps
#      preempt the build; the build still progresses, just slower.
#   3. `MAKEFLAGS="-j${QUIET_J:-2}"` — cap parallel C++ compile to 2
#      cores by default. The Idris elaboration phase is single-threaded
#      anyway, so capping the C++ phase leaves cores for the user
#      essentially for free on this workload. Override via env var if
#      you have a beefier box: `QUIET_J=4 scripts/perf-run-quiet.sh ...`.
#
# All other behaviour matches `perf-run.sh` byte-for-byte (this script
# just exec's into it after applying the three wrappers above): same
# arg forwarding, same `perf-log.jsonl` append, same exit-code
# propagation, same stdout summary.
#
# Usage:
#   scripts/perf-run-quiet.sh <example-key> <backend> [args...]
#
# Examples:
#   scripts/perf-run-quiet.sh hf-llama-generate torch
#   TORCH_DEVICE=mps scripts/perf-run-quiet.sh hf-llama-generate torch
#   MLX_DEVICE=gpu   scripts/perf-run-quiet.sh hf-llama-generate mlx
#   QUIET_J=4 TORCH_DEVICE=mps scripts/perf-run-quiet.sh hf-llama torch
#
# For the "give me full machine speed" case, use `perf-run.sh` directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
J="${QUIET_J:-2}"

# Build the inner command. `env MAKEFLAGS=...` rather than exporting
# globally so the parallelism cap is scoped to this invocation only.
INNER=(nice -n 19 env "MAKEFLAGS=-j${J}" "${SCRIPT_DIR}/perf-run.sh" "$@")

# caffeinate -i: prevent idle sleep. -i (idle) is the right knob — not
# -s (system), which is for daemons that own the wake reason. Available
# on macOS; pass-through on other systems.
if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
	exec caffeinate -i "${INNER[@]}"
else
	exec "${INNER[@]}"
fi

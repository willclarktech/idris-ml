#!/usr/bin/env bash
# scripts/perf-nightly.sh — Tier 2 perf gate (Axes A+B+C, tape only).
#
# Runs the Tier-1 op-kernel + layer benches via perf-fast.sh, then
# extends with Axis C (end-to-end training, one workload per training
# mode, capped to a short epoch count). Each Axis C workload runs the
# Idris example + matching PyTorch reference at the same fixed
# --epochs N --seed 42, greps the PERF_MS_PER_EP marker from each side,
# and appends a paired `kind: "op_bench"` JSONL entry with axis="C".
#
# Designed for a daily GitHub Actions schedule — total wall ≤ 20 min
# on tape. Per CLAUDE.md "all backends first-class" Tier 2 stays
# tape-only by design; Tier 3 (perf-sweep.sh) covers cross-backend.
#
# Schema (Axis C entries):
#   { "kind": "op_bench", "axis": "C", "section": "<example>-train",
#     "label": "<example>-train", "runtime": "tape" | "pytorch",
#     "wall_ms": ..., "iters": N, "ms_per_iter": ms_per_ep,
#     "commit": "...", "ts": "...", "date": "..." }
#
# Usage:
#   scripts/perf-nightly.sh            # full Tier 1 + Axis C
#   scripts/perf-nightly.sh --axis-c-only  # skip Tier 1
#
# Workloads (one per distinct training mode):
#   supervised   — feedforward
#   lstm         — recurrent
#   transformer  — sequence-to-sequence
#   ntm-copy     — memory / two-phase
#   reinforce    — RL on-policy

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_PATH="docs/develop/perf-log.jsonl"
SEED=42

DO_TIER1=1
for arg in "$@"; do
  case "$arg" in
    --axis-c-only) DO_TIER1=0 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "perf-nightly.sh: unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [ "$DO_TIER1" = "1" ]; then
  echo "==> perf-nightly: running Tier 1 (Axes A + B)"
  bash scripts/perf-fast.sh --no-render
fi

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [ -n "$(git status --porcelain -- ':!docs/develop/perf-log.jsonl' ':!BENCHMARKS.md' 2>/dev/null)" ]; then
  COMMIT="${COMMIT}+dirty"
fi
ISO_TS=$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')
DATE=$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%d"))')

# example-key  idris-make-target               idris-args-var  ref-module                      epochs
# Selected per the testing-taxonomy "one workload per training mode" rule.
# Epoch counts chosen to (a) get past startup-amortisation noise on the
# PERF_MS_PER_EP marker, (b) keep total Axis C wall under ~10 minutes.
AXIS_C_WORKLOADS=(
  "supervised  example-supervised  SUPERVISED_ARGS  torch_ref.scripts.supervised  50"
  "lstm        example-lstm        LSTM_ARGS        torch_ref.scripts.lstm        20"
  "transformer example-transformer TRANSFORMER_ARGS torch_ref.scripts.transformer 10"
  "ntm-copy    example-ntm-copy    NTM_COPY_ARGS    torch_ref.scripts.ntm_copy    10"
  "reinforce   example-reinforce   REINFORCE_ARGS   torch_ref.scripts.reinforce   50"
)

extract_marker() {
  local stdout_path="$1"
  local val
  val=$(grep -E '^PERF_MS_PER_EP=' "$stdout_path" | tail -1 | sed 's/^PERF_MS_PER_EP=//')
  if [ -z "$val" ]; then
    echo ""
  else
    echo "$val"
  fi
}

C_RESULTS=$(mktemp)
trap 'rm -f "$C_RESULTS"' EXIT

echo "==> perf-nightly: running Axis C (e2e training)"
for row in "${AXIS_C_WORKLOADS[@]}"; do
  read -r key idris_tgt idris_var ref_mod epochs <<< "$row"
  echo "  -> $key (epochs=$epochs)"

  IDRIS_OUT=$(mktemp)
  PY_OUT=$(mktemp)

  set +e
  caffeinate -i nice -n 19 env MAKEFLAGS=-j2 \
    make --no-print-directory "$idris_tgt" \
    "$idris_var=--epochs $epochs --seed $SEED" \
    >"$IDRIS_OUT" 2>&1
  set -e
  IDRIS_MS=$(extract_marker "$IDRIS_OUT")

  set +e
  ( cd packages/pytorch && uv run python -m "$ref_mod" \
      --epochs "$epochs" --seed "$SEED" ) >"$PY_OUT" 2>&1
  set -e
  PY_MS=$(extract_marker "$PY_OUT")

  echo "$key|$epochs|$IDRIS_MS|$PY_MS" >> "$C_RESULTS"
  rm -f "$IDRIS_OUT" "$PY_OUT"
done

echo "==> perf-nightly: appending Axis C entries to perf-log.jsonl"
COMMIT="$COMMIT" ISO_TS="$ISO_TS" DATE="$DATE" \
C_RESULTS="$C_RESULTS" LOG_PATH="$LOG_PATH" \
python3 <<'PY'
import json, os

iso_ts = os.environ["ISO_TS"]
date   = os.environ["DATE"]
commit = os.environ["COMMIT"]

count = 0
with open(os.environ["C_RESULTS"]) as fh, \
     open(os.environ["LOG_PATH"], "a") as out:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        key, epochs, idris_ms, py_ms = line.split("|")
        epochs = int(epochs)
        label = f"{key}-train"
        section = label  # one section per training-mode workload
        for runtime, ms in [("tape", idris_ms), ("pytorch", py_ms)]:
            if not ms:
                continue
            try:
                ms_f = float(ms)
            except ValueError:
                continue
            entry = {
                "ts": iso_ts,
                "date": date,
                "kind": "op_bench",
                "axis": "C",
                "section": section,
                "label": label,
                "runtime": runtime,
                "commit": commit,
                "wall_ms": ms_f * epochs,
                "iters": epochs,
                "ms_per_iter": ms_f,
            }
            out.write(json.dumps(entry) + "\n")
            count += 1
print(f"appended {count} axis-C entries to {os.environ['LOG_PATH']}")
PY

echo "==> perf-nightly: rendering BENCHMARKS.md"
python3 scripts/render-benchmarks.py

echo "==> perf-nightly: done"

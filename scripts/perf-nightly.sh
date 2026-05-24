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
#   scripts/perf-nightly.sh                # full Tier 1 + Axes C + D
#   scripts/perf-nightly.sh --axis-c-only  # skip Tier 1, skip Axis D
#   scripts/perf-nightly.sh --axis-d-only  # skip Tier 1, skip Axis C
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
DO_AXIS_C=1
DO_AXIS_D=1
for arg in "$@"; do
  case "$arg" in
    --axis-c-only) DO_TIER1=0; DO_AXIS_D=0 ;;
    --axis-d-only) DO_TIER1=0; DO_AXIS_C=0 ;;
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

# Axis D markers from HF inference examples — Idris-side emits a pair:
#   PERF_GENERATE_TOKENS=<N>
#   PERF_GENERATE_WALL_MS=<wall>
# Both sides (Idris + paired PyTorch ref) emit the same pair.
extract_axis_d_tokens() {
  local stdout_path="$1"
  grep -E '^PERF_GENERATE_TOKENS=' "$stdout_path" | tail -1 \
    | sed 's/^PERF_GENERATE_TOKENS=//'
}
extract_axis_d_wall() {
  local stdout_path="$1"
  grep -E '^PERF_GENERATE_WALL_MS=' "$stdout_path" | tail -1 \
    | sed 's/^PERF_GENERATE_WALL_MS=//'
}

C_RESULTS=$(mktemp)
D_RESULTS=$(mktemp)
trap 'rm -f "$C_RESULTS" "$D_RESULTS"' EXIT

if [ "$DO_AXIS_C" = "1" ]; then
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
fi

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

# AXIS D — HuggingFace inference.
# Each workload runs the Idris HF inference example + paired PyTorch
# script and parses two markers from each side:
#   PERF_GENERATE_TOKENS=<N>     — token-count denominator
#   PERF_GENERATE_WALL_MS=<wall> — wall around the inference window
# ms_per_iter = wall / tokens; iters = tokens.
#
# Idris-side make-target names match scripts/perf-run.sh's hf-* keys.
# Each example sets its own BACKEND/dtype via the Makefile recipe
# (typically tape F32 for bert/gpt2, torch-mps F32 for llama).
# Each row: key  idris-target  ref-module  idris-env (or "-" for none).
# Llama runs on torch-mps (F32 auto-set by the HF_GOALS branch in the
# Makefile) — tape can't hold a 1B model. On CI lanes without a
# torch/MPS build the Idris-side make will fail, the script skips
# the entry, and the BERT + GPT-2 entries still land.
AXIS_D_WORKLOADS=(
  "hf-bert            example-hf-bert-inference   torch_ref.scripts.hf_bert_inference  -"
  "hf-gpt2            example-hf-gpt2-inference   torch_ref.scripts.hf_gpt2_inference  -"
  "hf-llama-generate  example-hf-llama-inference  torch_ref.scripts.hf_llama_inference BACKEND=torch:TORCH_DEVICE=mps"
)

if [ "$DO_AXIS_D" = "1" ]; then
  echo "==> perf-nightly: running Axis D (HF inference)"
  for row in "${AXIS_D_WORKLOADS[@]}"; do
    read -r key idris_tgt ref_mod idris_env <<< "$row"
    echo "  -> $key"

    IDRIS_OUT=$(mktemp)
    PY_OUT=$(mktemp)

    # idris_env "-" means no per-workload env; otherwise it's a
    # colon-separated list of KEY=VALUE pairs (e.g. BACKEND=torch:TORCH_DEVICE=mps).
    env_args=()
    if [ "$idris_env" != "-" ]; then
      IFS=':' read -ra env_pairs <<< "$idris_env"
      for pair in "${env_pairs[@]}"; do
        env_args+=("$pair")
      done
    fi

    set +e
    caffeinate -i nice -n 19 env MAKEFLAGS=-j2 "${env_args[@]}" \
      make --no-print-directory "$idris_tgt" >"$IDRIS_OUT" 2>&1
    set -e
    IDRIS_TOK=$(extract_axis_d_tokens "$IDRIS_OUT")
    IDRIS_WALL=$(extract_axis_d_wall "$IDRIS_OUT")

    set +e
    ( cd packages/pytorch && uv run python -m "$ref_mod" ) >"$PY_OUT" 2>&1
    set -e
    PY_TOK=$(extract_axis_d_tokens "$PY_OUT")
    PY_WALL=$(extract_axis_d_wall "$PY_OUT")

    echo "$key|$IDRIS_TOK|$IDRIS_WALL|$PY_TOK|$PY_WALL" >> "$D_RESULTS"
    rm -f "$IDRIS_OUT" "$PY_OUT"
  done
fi

echo "==> perf-nightly: appending Axis D entries to perf-log.jsonl"
COMMIT="$COMMIT" ISO_TS="$ISO_TS" DATE="$DATE" \
D_RESULTS="$D_RESULTS" LOG_PATH="$LOG_PATH" \
python3 <<'PY'
import json, os

iso_ts = os.environ["ISO_TS"]
date   = os.environ["DATE"]
commit = os.environ["COMMIT"]

count = 0
with open(os.environ["D_RESULTS"]) as fh, \
     open(os.environ["LOG_PATH"], "a") as out:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        key, idris_tok, idris_wall, py_tok, py_wall = line.split("|")
        label = f"{key}-inference"
        section = label
        for runtime, tok, wall in [
            ("tape", idris_tok, idris_wall),
            ("pytorch", py_tok, py_wall),
        ]:
            if not tok or not wall:
                continue
            try:
                tok_i = int(tok)
                wall_f = float(wall)
            except ValueError:
                continue
            if tok_i <= 0:
                continue
            entry = {
                "ts": iso_ts,
                "date": date,
                "kind": "op_bench",
                "axis": "D",
                "section": section,
                "label": label,
                "runtime": runtime,
                "commit": commit,
                "wall_ms": wall_f,
                "iters": tok_i,
                "ms_per_iter": wall_f / tok_i,
            }
            out.write(json.dumps(entry) + "\n")
            count += 1
print(f"appended {count} axis-D entries to {os.environ['LOG_PATH']}")
PY

echo "==> perf-nightly: rendering BENCHMARKS.md"
python3 scripts/render-benchmarks.py

echo "==> perf-nightly: done"

#!/usr/bin/env bash
# scripts/perf-run.sh — run an example to convergence (or early-stop)
# and auto-log a structured entry into docs/develop/perf-log.jsonl.
#
# Each invocation appends a single JSON object on its own line to
# `perf-log.jsonl`. Schema and conventions are documented in
# `perf-log.md` (the companion markdown file).
#
# Usage:
#   scripts/perf-run.sh <example-key> <backend> [example-args...]
#
# Examples:
#   scripts/perf-run.sh ntm-copy tape --seed 42 --batch 1
#   scripts/perf-run.sh dnc-copy torch --seed 1
#
# Querying:
#   jq 'select(.example == "dnc-copy" and .backend == "tape")' \
#     docs/develop/perf-log.jsonl
#
# Per CLAUDE.md "Performance results" convention: every measurement
# is appended; never edit or delete prior entries. If a measurement
# is invalid, append a follow-up entry that says so.
set -euo pipefail

if [ $# -lt 2 ]; then
  cat <<EOF >&2
usage: $0 <example-key> <backend> [example-args...]

example-keys: ntm-copy, ntm-recall, dnc-copy, dnc-recall, supervised,
              rnn, lstm, gru, transformer, gpt, matmul-bench, mnist, seq-classify,
              reinforce, dqn, mountain-car, mountain-car-cont, a2c,
              ppo, sac, hf-bert, hf-gpt2, hf-llama, hf-llama-generate, hf-bitnet
backends:     tape, mlx, torch
EOF
  exit 2
fi

EXAMPLE_KEY="$1"; shift
BACKEND="$1"; shift
ARGS=("$@")

case "$EXAMPLE_KEY" in
  ntm-copy)            TGT=example-ntm-copy;                   AVAR=NTM_COPY_ARGS ;;
  ntm-recall)          TGT=example-ntm-associative-recall;     AVAR=NTM_RECALL_ARGS ;;
  dnc-copy)            TGT=example-dnc-copy;                   AVAR=DNC_COPY_ARGS ;;
  dnc-recall)          TGT=example-dnc-recall;                 AVAR=DNC_RECALL_ARGS ;;
  supervised)          TGT=example-supervised;                 AVAR=SUPERVISED_ARGS ;;
  rnn)                 TGT=example-rnn;                        AVAR=RNN_ARGS ;;
  lstm)                TGT=example-lstm;                       AVAR=LSTM_ARGS ;;
  gru)                 TGT=example-gru;                        AVAR=GRU_ARGS ;;
  transformer)         TGT=example-transformer;                AVAR=TRANSFORMER_ARGS ;;
  gpt)                 TGT=example-gpt;                        AVAR=GPT_ARGS ;;
  matmul-bench)        TGT=example-matmul-bench;               AVAR=MATMUL_BENCH_ARGS ;;
  mnist)               TGT=example-mnist;                      AVAR=MNIST_ARGS ;;
  seq-classify)        TGT=example-seq-classify;               AVAR=SEQ_ARGS ;;
  reinforce)           TGT=example-reinforce;                  AVAR=REINFORCE_ARGS ;;
  dqn)                 TGT=example-dqn;                        AVAR=DQN_ARGS ;;
  mountain-car)        TGT=example-mountain-car;               AVAR=MOUNTAIN_CAR_ARGS ;;
  mountain-car-cont)   TGT=example-mountain-car-cont;          AVAR=MOUNTAIN_CAR_CONT_ARGS ;;
  a2c)                 TGT=example-a2c;                        AVAR=A2C_ARGS ;;
  ppo)                 TGT=example-ppo;                        AVAR=PPO_ARGS ;;
  sac)                 TGT=example-sac;                        AVAR=SAC_ARGS ;;
  # HF inference examples — no training loop, no RESULT line; we extract
  # `[stage] [hh:mm:ss] <label>` timings into entry.stages instead. AVAR
  # is set to a no-op make-variable name so the existing AVAR=ARGS
  # plumbing doesn't fight us (the inference examples don't take CLI args
  # via *_ARGS).
  hf-bert)             TGT=example-hf-bert-inference;          AVAR=_HF_NOARGS ;;
  hf-gpt2)             TGT=example-hf-gpt2-inference;          AVAR=_HF_NOARGS ;;
  hf-llama)            TGT=example-hf-llama-inference;         AVAR=_HF_NOARGS ;;
  # Same example as hf-llama, but with the --dump-tokens / multi-step
  # generation gate path exercised. Distinguished from `hf-llama` so
  # the perf-log entry carries the gate's wall-clock separately from
  # the user-facing-demo wall-clock (they decode for different default
  # budgets; the gate is fixed at --num-tokens 4 in the Makefile).
  hf-llama-generate)   TGT=test-hf-llama-generate-roundtrip;   AVAR=_HF_NOARGS ;;
  hf-bitnet)           TGT=example-hf-bitnet-inference;        AVAR=_HF_NOARGS ;;
  *) echo "unknown example-key: $EXAMPLE_KEY" >&2; exit 2 ;;
esac

# Commit metadata at run-time. +dirty if the working tree has uncommitted
# changes — important so we know the entry isn't reproducible from a
# clean checkout. perf-log.jsonl is excluded from the dirty check since
# this script itself appends to it mid-run; including it would mark
# every entry after the first as +dirty.
COMMIT=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- ':!docs/develop/perf-log.jsonl' 2>/dev/null)" ]; then
  COMMIT="${COMMIT}+dirty"
fi
DATE=$(date +%Y-%m-%d)

# Device of record. Both mlx and torch support a runtime device switch
# (MLX_DEVICE=cpu|gpu; TORCH_DEVICE=cpu|mps|cuda). tape is C-on-CPU.
case "$BACKEND" in
  mlx)   DEVICE="${MLX_DEVICE:-cpu}" ;;
  tape)  DEVICE="cpu" ;;
  torch) DEVICE="${TORCH_DEVICE:-cpu}" ;;
  *)     DEVICE="unknown" ;;
esac
# mlx accepts "metal" as a synonym for "gpu"; normalize.
[ "$DEVICE" = "metal" ] && DEVICE="gpu"

# Dtype override of record. Only set when caller explicitly chose
# TORCH_DTYPE / MLX_DTYPE / TAPE_DTYPE; empty otherwise (the
# BuildConfig default for the (backend, device) cell applies — F32 for
# torch-mps and mlx-gpu, F64 elsewhere). Tracked in the JSONL entry so
# a BF16/F16 run is visibly distinct from the default-F32 run on the
# same example/backend/device.
TORCH_DTYPE_STATE="${TORCH_DTYPE:-}"
MLX_DTYPE_STATE="${MLX_DTYPE:-}"
TAPE_DTYPE_STATE="${TAPE_DTYPE:-}"

# MLX_COMPILE state of record (Job 3 Phase B opt-in). Only meaningful
# on the mlx backend; non-mlx always records "n/a".
case "$BACKEND" in
  mlx)
    case "${MLX_COMPILE:-}" in
      1|true|yes) MLX_COMPILE_STATE="on" ;;
      *)          MLX_COMPILE_STATE="off" ;;
    esac
    ;;
  *) MLX_COMPILE_STATE="n/a" ;;
esac

# Run the example and capture output. Use stdbuf -oL so we can tail
# the log live during long-running tasks (per the "always use stdbuf"
# convention for background tasks).
LOG=$(mktemp)
T0=$(python3 -c 'import time; print(int(time.time_ns()/1_000_000))')
set +e
BACKEND="$BACKEND" stdbuf -oL make --no-print-directory "$TGT" \
  "${AVAR}=${ARGS[*]}" > "$LOG" 2>&1
RC=$?
set -e
T1=$(python3 -c 'import time; print(int(time.time_ns()/1_000_000))')

ELAPSED_MS=$((T1 - T0))
ELAPSED_PRETTY=$(python3 -c "
ms = $ELAPSED_MS
s = ms // 1000
m = s // 60
s = s % 60
h = m // 60
m = m % 60
if h > 0:
    print(f'{h}h {m}m {s}s')
elif m > 0:
    print(f'{m}m {s}s')
else:
    print(f'{s}.{ms%1000:03d}s')
")

# Extract the canonical lines
RESULT_LINE=$(grep '^RESULT' "$LOG" | tail -1 || true)
COMPLETED_LINE=$(grep '^Completed' "$LOG" | tail -1 || true)
CONVERGED_LINE=$(grep -E '^\s*\[[^]]+\]\s+Converged' "$LOG" | tail -1 || true)
DIVERGED_LINE=$(grep -E '^\s*\[[^]]+\]\s+Diverged' "$LOG" | tail -1 || true)
# HF inference examples emit per-stage timings as
# `[stage] [hh:mm:ss] <label>` (e.g. `[stage] [00:00:22] hfLlamaModel ok`).
# Capture all such lines so the JSON entry's `stages` field reflects
# per-phase wall (state construction vs load vs decode), not just the
# total. Empty for training-loop examples.
STAGE_LINES=$(grep -E '^\[stage\] \[[0-9]{2}:[0-9]{2}:[0-9]{2}\]' "$LOG" || true)
# HF inference examples may also emit `[perf] step N: K ops` lines from
# the per-forward op-submission counter (TODO #393 diagnostic harness).
# Surface these alongside stages so the operator sees per-step counts;
# the JSON entry doesn't include them (they're per-forward, not
# per-phase). Empty for non-instrumented examples.
PERF_LINES=$(grep -E '^\[perf\]' "$LOG" || true)

LOG_PATH="docs/develop/perf-log.jsonl"
if [ ! -e "$LOG_PATH" ]; then
  : > "$LOG_PATH"
fi

# Build the JSON entry via python (bash JSON construction is misery —
# values may contain special characters that need escaping).
ARG_SUMMARY="${ARGS[*]:-defaults}"
ISO_TS=$(python3 -c 'import time; from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')

JSON_LINE=$(
  ARG_SUMMARY="$ARG_SUMMARY" \
  ISO_TS="$ISO_TS" DATE="$DATE" EXAMPLE_KEY="$EXAMPLE_KEY" \
  BACKEND="$BACKEND" DEVICE="$DEVICE" MLX_COMPILE_STATE="$MLX_COMPILE_STATE" \
  TORCH_DTYPE_STATE="$TORCH_DTYPE_STATE" \
  MLX_DTYPE_STATE="$MLX_DTYPE_STATE" \
  TAPE_DTYPE_STATE="$TAPE_DTYPE_STATE" \
  COMMIT="$COMMIT" RC="$RC" \
  ELAPSED_MS="$ELAPSED_MS" ELAPSED_PRETTY="$ELAPSED_PRETTY" \
  CONVERGED_LINE="$CONVERGED_LINE" DIVERGED_LINE="$DIVERGED_LINE" \
  COMPLETED_LINE="$COMPLETED_LINE" RESULT_LINE="$RESULT_LINE" \
  STAGE_LINES="$STAGE_LINES" \
  python3 <<'PY'
import json, os, re

# Pull the convergence epoch out of the converged/diverged line if present.
def parse_epoch(line):
    if not line:
        return None
    m = re.search(r"epoch\s+(\d+)", line)
    return int(m.group(1)) if m else None

# Parse "Completed in 1m 7s (5500 epochs, 12ms/epoch)" → ms_per_epoch + epochs.
def parse_completed(line):
    if not line:
        return {}
    out = {}
    m = re.search(r"\((\d+)\s+epochs?,\s+([\d.]+)\s*ms/epoch\)", line)
    if m:
        out["total_epochs"] = int(m.group(1))
        out["ms_per_epoch"] = float(m.group(2))
    m = re.search(r"Completed in\s+([^()]+)\s+\(", line)
    if m:
        out["wall"] = m.group(1).strip()
    return out

# Parse "RESULT\tepochs=5500\tacc_short=0.99..." → dict of fields.
def parse_result(line):
    if not line:
        return {}
    out = {}
    for part in line.split("\t")[1:]:  # skip leading "RESULT"
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out

# Parse multi-line `[stage] [hh:mm:ss] <label>` block → list of
# {"label": str, "elapsed_s": int} entries. Cumulative wall (seconds
# since program start), not deltas — matches the raw `[hh:mm:ss]` in
# the log. Caller can compute deltas if wanted. Returns [] when no
# stage lines were captured.
def parse_stages(blob):
    if not blob:
        return []
    out = []
    for line in blob.splitlines():
        m = re.match(r"^\[stage\] \[(\d{2}):(\d{2}):(\d{2})\]\s+(.*)$", line)
        if not m:
            continue
        h, mi, s, label = m.groups()
        elapsed = int(h) * 3600 + int(mi) * 60 + int(s)
        out.append({"label": label.strip(), "elapsed_s": elapsed})
    return out

entry = {
    "ts": os.environ["ISO_TS"],
    "date": os.environ["DATE"],
    "kind": "run",
    "example": os.environ["EXAMPLE_KEY"],
    "backend": os.environ["BACKEND"],
    "device": os.environ["DEVICE"],
    "mlx_compile": os.environ["MLX_COMPILE_STATE"],
    "commit": os.environ["COMMIT"],
    "args": os.environ["ARG_SUMMARY"],
    "exit": int(os.environ["RC"]),
    "wall_ms": int(os.environ["ELAPSED_MS"]),
    "wall_human": os.environ["ELAPSED_PRETTY"],
}
# Only emit *_dtype when explicitly set; absence means "BuildConfig
# default for the (backend, device) cell" (F32 for torch-mps and
# mlx-gpu, F64 elsewhere).
if os.environ.get("TORCH_DTYPE_STATE"):
    entry["torch_dtype"] = os.environ["TORCH_DTYPE_STATE"]
if os.environ.get("MLX_DTYPE_STATE"):
    entry["mlx_dtype"] = os.environ["MLX_DTYPE_STATE"]
if os.environ.get("TAPE_DTYPE_STATE"):
    entry["tape_dtype"] = os.environ["TAPE_DTYPE_STATE"]
conv = parse_epoch(os.environ.get("CONVERGED_LINE", ""))
div  = parse_epoch(os.environ.get("DIVERGED_LINE",  ""))
if conv is not None: entry["converged_at_epoch"] = conv
if div  is not None: entry["diverged_at_epoch"]  = div
stats = parse_completed(os.environ.get("COMPLETED_LINE", ""))
if stats: entry["stats"] = stats
result = parse_result(os.environ.get("RESULT_LINE", ""))
if result: entry["result"] = result
stages = parse_stages(os.environ.get("STAGE_LINES", ""))
if stages: entry["stages"] = stages

print(json.dumps(entry))
PY
)

echo "$JSON_LINE" >> "$LOG_PATH"

# Mirror to stdout for the operator
DTYPE_TAG="${TORCH_DTYPE_STATE}${MLX_DTYPE_STATE}${TAPE_DTYPE_STATE}"
echo "=== ${EXAMPLE_KEY} [${BACKEND}/${DEVICE}${DTYPE_TAG:+/$DTYPE_TAG}] @ ${COMMIT} ==="
echo "wall:    ${ELAPSED_PRETTY} (exit ${RC})"
[ -n "$CONVERGED_LINE" ] && echo "${CONVERGED_LINE}"
[ -n "$DIVERGED_LINE"  ] && echo "${DIVERGED_LINE}"
[ -n "$COMPLETED_LINE" ] && echo "${COMPLETED_LINE}"
[ -n "$RESULT_LINE"    ] && echo "${RESULT_LINE}"
[ -n "$STAGE_LINES"    ] && printf '%s\n' "$STAGE_LINES"
[ -n "$PERF_LINES"     ] && printf '%s\n' "$PERF_LINES"
echo "Logged to ${LOG_PATH}"

# Forward the make exit code so callers can detect failure.
exit $RC

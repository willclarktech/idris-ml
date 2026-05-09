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
              rnn, lstm, gru, transformer, gpt, mnist, seq-classify,
              reinforce, dqn, mountain-car, mountain-car-cont, a2c,
              ppo, sac
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
  mnist)               TGT=example-mnist;                      AVAR=MNIST_ARGS ;;
  seq-classify)        TGT=example-seq-classify;               AVAR=SEQ_ARGS ;;
  reinforce)           TGT=example-reinforce;                  AVAR=REINFORCE_ARGS ;;
  dqn)                 TGT=example-dqn;                        AVAR=DQN_ARGS ;;
  mountain-car)        TGT=example-mountain-car;               AVAR=MOUNTAIN_CAR_ARGS ;;
  mountain-car-cont)   TGT=example-mountain-car-cont;          AVAR=MOUNTAIN_CAR_CONT_ARGS ;;
  a2c)                 TGT=example-a2c;                        AVAR=A2C_ARGS ;;
  ppo)                 TGT=example-ppo;                        AVAR=PPO_ARGS ;;
  sac)                 TGT=example-sac;                        AVAR=SAC_ARGS ;;
  *) echo "unknown example-key: $EXAMPLE_KEY" >&2; exit 2 ;;
esac

# Commit metadata at run-time. +dirty if the working tree has uncommitted
# changes — important so we know the entry isn't reproducible from a
# clean checkout.
COMMIT=$(git rev-parse --short HEAD)
if ! git diff --quiet || ! git diff --cached --quiet; then
  COMMIT="${COMMIT}+dirty"
fi
DATE=$(date +%Y-%m-%d)

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
  BACKEND="$BACKEND" COMMIT="$COMMIT" RC="$RC" \
  ELAPSED_MS="$ELAPSED_MS" ELAPSED_PRETTY="$ELAPSED_PRETTY" \
  CONVERGED_LINE="$CONVERGED_LINE" DIVERGED_LINE="$DIVERGED_LINE" \
  COMPLETED_LINE="$COMPLETED_LINE" RESULT_LINE="$RESULT_LINE" \
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

entry = {
    "ts": os.environ["ISO_TS"],
    "date": os.environ["DATE"],
    "kind": "run",
    "example": os.environ["EXAMPLE_KEY"],
    "backend": os.environ["BACKEND"],
    "commit": os.environ["COMMIT"],
    "args": os.environ["ARG_SUMMARY"],
    "exit": int(os.environ["RC"]),
    "wall_ms": int(os.environ["ELAPSED_MS"]),
    "wall_human": os.environ["ELAPSED_PRETTY"],
}
conv = parse_epoch(os.environ.get("CONVERGED_LINE", ""))
div  = parse_epoch(os.environ.get("DIVERGED_LINE",  ""))
if conv is not None: entry["converged_at_epoch"] = conv
if div  is not None: entry["diverged_at_epoch"]  = div
stats = parse_completed(os.environ.get("COMPLETED_LINE", ""))
if stats: entry["stats"] = stats
result = parse_result(os.environ.get("RESULT_LINE", ""))
if result: entry["result"] = result

print(json.dumps(entry))
PY
)

echo "$JSON_LINE" >> "$LOG_PATH"

# Mirror to stdout for the operator
echo "=== ${EXAMPLE_KEY} [${BACKEND}] @ ${COMMIT} ==="
echo "wall:    ${ELAPSED_PRETTY} (exit ${RC})"
[ -n "$CONVERGED_LINE" ] && echo "${CONVERGED_LINE}"
[ -n "$DIVERGED_LINE"  ] && echo "${DIVERGED_LINE}"
[ -n "$COMPLETED_LINE" ] && echo "${COMPLETED_LINE}"
[ -n "$RESULT_LINE"    ] && echo "${RESULT_LINE}"
echo "Logged to ${LOG_PATH}"

# Forward the make exit code so callers can detect failure.
exit $RC

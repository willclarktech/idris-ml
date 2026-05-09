#!/usr/bin/env bash
# scripts/perf-run.sh — run an example to convergence (or early-stop)
# and auto-log a structured entry into docs/develop/perf-log.md.
#
# Captures: date, short commit hash (with +dirty suffix if uncommitted
# changes), example, backend, full args, RESULT line, "Completed in X
# epochs" line, total wall-clock, and any "Converged at epoch N"
# marker.
#
# Usage:
#   scripts/perf-run.sh <example-key> <backend> [example-args...]
#
# Examples:
#   scripts/perf-run.sh ntm-copy tape --seed 42 --batch 1
#   scripts/perf-run.sh dnc-copy torch --seed 1
#
# Per CLAUDE.md "Performance results" convention: every measurement
# goes into perf-log.md with its commit hash. Never edit or delete
# prior entries; if a measurement is invalid, append a follow-up
# entry that says so.
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

LOG_PATH="docs/develop/perf-log.md"
if [ ! -f "$LOG_PATH" ]; then
  echo "ERROR: $LOG_PATH does not exist" >&2
  exit 1
fi

# Compose the entry. The header self-describes (example, backend,
# commit, args summary) so readers can grep `^### ` for a chronological
# scan, or grep `ntm-copy.*tape` to filter by example and backend.
# Don't try to insert into bucketed sections — append at end and let
# the header carry the metadata. That keeps the file strictly
# append-only.
ARG_SUMMARY="${ARGS[*]:-defaults}"
ENTRY=$(cat <<EOF

### ${DATE} — \`${EXAMPLE_KEY}\` [${BACKEND}] @ \`${COMMIT}\` — \`${ARG_SUMMARY}\`

exit:    ${RC}
wall:    ${ELAPSED_PRETTY} (${ELAPSED_MS} ms)
${CONVERGED_LINE:+converged: ${CONVERGED_LINE#*\] }}${CONVERGED_LINE:+
}${DIVERGED_LINE:+diverged: ${DIVERGED_LINE#*\] }}${DIVERGED_LINE:+
}${COMPLETED_LINE:+stats:   ${COMPLETED_LINE}}${COMPLETED_LINE:+
}${RESULT_LINE:+result:  \`${RESULT_LINE}\`}
EOF
)

echo "$ENTRY" >> "$LOG_PATH"

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

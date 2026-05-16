#!/usr/bin/env bash
# scripts/perf-baseline.sh
#
# Measure ms/epoch for an example on a given backend, plus the matching
# PyTorch reference. Output one CSV row to stdout:
#
#   example,backend,idris_ms_per_epoch,pytorch_ms_per_epoch,ratio,epochs,seed,commit,notes
#
# `commit` is the abbreviated git hash of HEAD, with `+dirty` appended
# if the working tree has uncommitted changes (mirrors perf-run.sh).
#
# Usage:
#   scripts/perf-baseline.sh <example-key> <backend>
#
# Examples (key -> idris-target -> ref-script -> CLI args):
#   See EXAMPLES table below. Each row maps a friendly key to:
#     - the make target
#     - the args var name (e.g. NTM_COPY_ARGS)
#     - the python ref module
#     - the per-example budget (epochs)
#
# Notes:
#  - Wall-clock is total (build + run + teardown). For fair ms/epoch we
#    do TWO runs at different epoch counts and subtract: timing(N) -
#    timing(N/4) ≈ (3/4)*N timed epochs, removing fixed overhead. This is
#    cheap when both runs share the same idris2 build cache.
#  - The PyTorch side has effectively no build cost; one run per example.
#  - All idris-side runs use BACKEND=$2 from the env.
#  - All runs use --seed 42.
set -euo pipefail

EXAMPLE_KEY="${1:-}"
BACKEND="${2:-tape}"
SEED=42

if [ -z "$EXAMPLE_KEY" ]; then
  echo "usage: $0 <example-key> <backend>"
  echo "  example-keys: see source"
  exit 2
fi

# example-key -> (idris-make-target, idris-args-var, ref-py-module, n_short, n_long)
case "$EXAMPLE_KEY" in
  supervised)   IDRIS_TGT=example-supervised;        IDRIS_VAR=SUPERVISED_ARGS;     REF_MOD=torch_ref.scripts.supervised;        N_SHORT=50;  N_LONG=200 ;;
  rnn)          IDRIS_TGT=example-rnn;               IDRIS_VAR=RNN_ARGS;            REF_MOD=torch_ref.scripts.rnn;               N_SHORT=50;  N_LONG=200 ;;
  lstm)         IDRIS_TGT=example-lstm;              IDRIS_VAR=LSTM_ARGS;           REF_MOD=torch_ref.scripts.lstm;              N_SHORT=50;  N_LONG=200 ;;
  gru)          IDRIS_TGT=example-gru;               IDRIS_VAR=GRU_ARGS;            REF_MOD=torch_ref.scripts.gru;               N_SHORT=50;  N_LONG=200 ;;
  transformer)  IDRIS_TGT=example-transformer;       IDRIS_VAR=TRANSFORMER_ARGS;    REF_MOD=torch_ref.scripts.transformer;       N_SHORT=50;  N_LONG=200 ;;
  ntm-copy)     IDRIS_TGT=example-ntm-copy;          IDRIS_VAR=NTM_COPY_ARGS;       REF_MOD=torch_ref.scripts.ntm_copy;          N_SHORT=10;  N_LONG=40  ;;
  ntm-recall)   IDRIS_TGT=example-ntm-associative-recall; IDRIS_VAR=NTM_RECALL_ARGS; REF_MOD=torch_ref.scripts.ntm_recall;       N_SHORT=10;  N_LONG=40  ;;
  dnc-copy)     IDRIS_TGT=example-dnc-copy;          IDRIS_VAR=DNC_COPY_ARGS;       REF_MOD=torch_ref.scripts.dnc_copy;          N_SHORT=20;  N_LONG=80  ;;
  dnc-recall)   IDRIS_TGT=example-dnc-recall;        IDRIS_VAR=DNC_RECALL_ARGS;     REF_MOD=torch_ref.scripts.dnc_recall;        N_SHORT=10;  N_LONG=40  ;;
  reinforce)    IDRIS_TGT=example-reinforce;         IDRIS_VAR=REINFORCE_ARGS;      REF_MOD=torch_ref.scripts.reinforce;         N_SHORT=50;  N_LONG=200 ;;
  dqn)          IDRIS_TGT=example-dqn;               IDRIS_VAR=DQN_ARGS;            REF_MOD=torch_ref.scripts.dqn;               N_SHORT=20;  N_LONG=80  ;;
  mountain-car) IDRIS_TGT=example-mountain-car;      IDRIS_VAR=MOUNTAIN_CAR_ARGS;   REF_MOD=torch_ref.scripts.mountain_car;      N_SHORT=20;  N_LONG=80  ;;
  mountain-car-cont) IDRIS_TGT=example-mountain-car-cont; IDRIS_VAR=MOUNTAIN_CAR_CONT_ARGS; REF_MOD=torch_ref.scripts.mountain_car_cont; N_SHORT=20; N_LONG=80 ;;
  a2c)          IDRIS_TGT=example-a2c;               IDRIS_VAR=A2C_ARGS;            REF_MOD=torch_ref.scripts.a2c;               N_SHORT=50;  N_LONG=200 ;;
  ppo)          IDRIS_TGT=example-ppo;               IDRIS_VAR=PPO_ARGS;            REF_MOD=torch_ref.scripts.ppo;               N_SHORT=10;  N_LONG=40  ;;
  sac)          IDRIS_TGT=example-sac;               IDRIS_VAR=SAC_ARGS;            REF_MOD=torch_ref.scripts.sac;               N_SHORT=20;  N_LONG=80  ;;
  *)
    echo "unknown example-key: $EXAMPLE_KEY" >&2
    exit 2 ;;
esac

# Run idris example with `--epochs N --seed S`, return total wall-clock in ms.
# Uses time.time_ns() (absolute since epoch) — time.monotonic_ns() is
# process-relative, so its value resets each `python3 -c` invocation and
# the diff is meaningless.
run_idris() {
  local n="$1"
  local t0 t1
  t0=$(python3 -c 'import time; print(int(time.time_ns()/1_000_000))')
  BACKEND="$BACKEND" make --no-print-directory "$IDRIS_TGT" \
    "$IDRIS_VAR=--epochs $n --seed $SEED" >/dev/null 2>&1
  t1=$(python3 -c 'import time; print(int(time.time_ns()/1_000_000))')
  echo $((t1 - t0))
}

# Same for pytorch ref.
run_pytorch() {
  local n="$1"
  local t0 t1
  t0=$(python3 -c 'import time; print(int(time.time_ns()/1_000_000))')
  ( cd packages/pytorch && uv run python -m "$REF_MOD" --epochs "$n" --seed "$SEED" ) >/dev/null 2>&1
  t1=$(python3 -c 'import time; print(int(time.time_ns()/1_000_000))')
  echo $((t1 - t0))
}

# Two-point timing: ms_per_epoch ≈ (T_long - T_short) / (N_long - N_short).
# T_short captures fixed overhead (build / startup / teardown).
#
# Warmup pass first so that idris2 compile cache, dylib paths, etc. are
# all warm — otherwise the first timed run pays cold-cache cost and the
# subtraction goes negative.
two_point_idris() {
  local t_short t_long
  run_idris "$N_SHORT" >/dev/null   # warmup
  t_short=$(run_idris "$N_SHORT")
  t_long=$(run_idris "$N_LONG")
  python3 -c "print(round((${t_long} - ${t_short}) / (${N_LONG} - ${N_SHORT}), 2))"
}

two_point_pytorch() {
  local t_short t_long
  run_pytorch "$N_SHORT" >/dev/null   # warmup
  t_short=$(run_pytorch "$N_SHORT")
  t_long=$(run_pytorch "$N_LONG")
  python3 -c "print(round((${t_long} - ${t_short}) / (${N_LONG} - ${N_SHORT}), 2))"
}

# Capture the active commit (with +dirty marker), mirroring perf-run.sh.
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  COMMIT="${COMMIT}+dirty"
fi

echo "[perf-baseline] $EXAMPLE_KEY [$BACKEND]: idris N=$N_SHORT then $N_LONG..." >&2
IDRIS_MS=$(two_point_idris)
echo "[perf-baseline] $EXAMPLE_KEY: pytorch N=$N_SHORT then $N_LONG..." >&2
PY_MS=$(two_point_pytorch)
RATIO=$(python3 -c "print(round(${IDRIS_MS} / ${PY_MS}, 2) if ${PY_MS} > 0 else 'inf')")

echo "${EXAMPLE_KEY},${BACKEND},${IDRIS_MS},${PY_MS},${RATIO},${N_LONG},${SEED},${COMMIT},"

# Also append a JSONL entry to perf-log.jsonl. Tagged kind="baseline"
# (vs perf-run.sh's default kind="convergence") so jq can filter:
#   jq 'select(.kind=="baseline")' docs/develop/perf-log.jsonl
LOG_PATH="docs/develop/perf-log.jsonl"
if [ ! -e "$LOG_PATH" ]; then : > "$LOG_PATH"; fi
ISO_TS=$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')
DATE=$(date +%Y-%m-%d)
case "$BACKEND" in
  mlx)   DEVICE="${MLX_DEVICE:-cpu}" ;;
  tape)  DEVICE="cpu" ;;
  torch) DEVICE="cpu" ;;
  *)     DEVICE="unknown" ;;
esac
[ "$DEVICE" = "metal" ] && DEVICE="gpu"
ISO_TS="$ISO_TS" DATE="$DATE" EXAMPLE_KEY="$EXAMPLE_KEY" BACKEND="$BACKEND" \
  DEVICE="$DEVICE" COMMIT="$COMMIT" IDRIS_MS="$IDRIS_MS" PY_MS="$PY_MS" RATIO="$RATIO" \
  N_LONG="$N_LONG" SEED="$SEED" \
  python3 <<'PY' >> "$LOG_PATH"
import json, os
def num(s):
    try: return float(s)
    except (ValueError, TypeError): return None
print(json.dumps({
    "ts": os.environ["ISO_TS"],
    "date": os.environ["DATE"],
    "kind": "baseline",
    "example": os.environ["EXAMPLE_KEY"],
    "backend": os.environ["BACKEND"],
    "device": os.environ["DEVICE"],
    "commit": os.environ["COMMIT"],
    "idris_ms_per_epoch": num(os.environ["IDRIS_MS"]),
    "pytorch_ms_per_epoch": num(os.environ["PY_MS"]),
    "ratio": num(os.environ["RATIO"]),
    "n_long": int(os.environ["N_LONG"]),
    "seed": int(os.environ["SEED"]),
}))
PY

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
#  - Timing methodology: both sides print a `PERF_MS_PER_EP=<float>`
#    marker after the training loop, before eval. The script greps the
#    marker from stdout. Eliminates startup / build / eval / Python
#    import variance from the measurement — the timer is *inside* the
#    training process. Previous version did wall-clock two-point
#    subtraction, which collapsed for short-converging RL refs (the
#    signal was below the startup-variance noise floor; see
#    `docs/develop/perf-changes.md` 2026-05-19 entry).
#  - Each side runs once at N=$N_LONG. Marker emit-points:
#      Idris:   `Train.runTrainingIO` → `formatPerfMsPerEp` (Util.idr)
#      PyTorch: `runner.run_training` for non-RL refs; per-script for RL.
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

# example-key -> (idris-make-target, idris-args-var, ref-py-module, n_long)
# N_LONG is the single epoch count passed to both sides. Each side
# emits PERF_MS_PER_EP after its training loop; the marker comes
# from inside the training process and is already a per-epoch number.
case "$EXAMPLE_KEY" in
  supervised)        IDRIS_TGT=example-supervised;             IDRIS_VAR=SUPERVISED_ARGS;        REF_MOD=torch_ref.scripts.supervised;        N_LONG=200  ;;
  rnn)               IDRIS_TGT=example-rnn;                    IDRIS_VAR=RNN_ARGS;               REF_MOD=torch_ref.scripts.rnn;               N_LONG=200  ;;
  lstm)              IDRIS_TGT=example-lstm;                   IDRIS_VAR=LSTM_ARGS;              REF_MOD=torch_ref.scripts.lstm;              N_LONG=200  ;;
  gru)               IDRIS_TGT=example-gru;                    IDRIS_VAR=GRU_ARGS;               REF_MOD=torch_ref.scripts.gru;               N_LONG=200  ;;
  transformer)       IDRIS_TGT=example-transformer;            IDRIS_VAR=TRANSFORMER_ARGS;       REF_MOD=torch_ref.scripts.transformer;       N_LONG=200  ;;
  ntm-copy)          IDRIS_TGT=example-ntm-copy;               IDRIS_VAR=NTM_COPY_ARGS;          REF_MOD=torch_ref.scripts.ntm_copy;          N_LONG=40   ;;
  ntm-recall)        IDRIS_TGT=example-ntm-associative-recall; IDRIS_VAR=NTM_ASSOCIATIVE_RECALL_ARGS;        REF_MOD=torch_ref.scripts.ntm_recall;        N_LONG=40   ;;
  dnc-copy)          IDRIS_TGT=example-dnc-copy;               IDRIS_VAR=DNC_COPY_ARGS;          REF_MOD=torch_ref.scripts.dnc_copy;          N_LONG=80   ;;
  dnc-recall)        IDRIS_TGT=example-dnc-recall;             IDRIS_VAR=DNC_RECALL_ARGS;        REF_MOD=torch_ref.scripts.dnc_recall;        N_LONG=40   ;;
  reinforce)         IDRIS_TGT=example-reinforce;              IDRIS_VAR=REINFORCE_ARGS;         REF_MOD=torch_ref.scripts.reinforce;         N_LONG=200  ;;
  dqn)               IDRIS_TGT=example-dqn;                    IDRIS_VAR=DQN_ARGS;               REF_MOD=torch_ref.scripts.dqn;               N_LONG=80   ;;
  mountain-car)      IDRIS_TGT=example-mountain-car;           IDRIS_VAR=MOUNTAIN_CAR_ARGS;      REF_MOD=torch_ref.scripts.mountain_car;      N_LONG=80   ;;
  mountain-car-cont) IDRIS_TGT=example-mountain-car-cont;      IDRIS_VAR=MOUNTAIN_CAR_CONT_ARGS; REF_MOD=torch_ref.scripts.mountain_car_cont; N_LONG=2000 ;;
  a2c)               IDRIS_TGT=example-a2c;                    IDRIS_VAR=A2C_ARGS;               REF_MOD=torch_ref.scripts.a2c;               N_LONG=200  ;;
  ppo)               IDRIS_TGT=example-ppo;                    IDRIS_VAR=PPO_ARGS;               REF_MOD=torch_ref.scripts.ppo;               N_LONG=40   ;;
  sac)               IDRIS_TGT=example-sac;                    IDRIS_VAR=SAC_ARGS;               REF_MOD=torch_ref.scripts.sac;               N_LONG=2000 ;;
  *)
    echo "unknown example-key: $EXAMPLE_KEY" >&2
    exit 2 ;;
esac

# Build the example binary ONCE so `run_idris` can invoke it directly
# without paying make's per-call overhead (dependency check, dylib
# copy, fork — ~50-500 ms variable, larger than the per-epoch signal
# on tiny tape examples like rnn/gru). The recipe always runs the
# binary at the end; we pass `--epochs 1` so the post-build run is
# cheap and serves as a warmup we discard.
build_idris_binary() {
  BACKEND="$BACKEND" make --no-print-directory "$IDRIS_TGT" \
    "$IDRIS_VAR=--epochs 1 --seed $SEED" >/dev/null 2>&1 || true
}

# Derive binary path from make target: example-rnn -> ./build/exec/rnn.
binary_for_target() {
  echo "./build/exec/${IDRIS_TGT#example-}"
}

# Extract `PERF_MS_PER_EP=<float>` from stdout. Returns the float as
# text, or "missing" if no marker was printed. Both Idris (via
# Util.formatPerfMsPerEp) and PyTorch (via runner.run_training plus
# per-RL-script print) emit exactly one such line.
extract_marker() {
  local stdout_path="$1"
  local val
  val=$(grep -E '^PERF_MS_PER_EP=' "$stdout_path" | tail -1 | sed 's/^PERF_MS_PER_EP=//')
  if [ -z "$val" ]; then
    echo "missing"
  else
    # Normalize to 2 decimals for output; preserve full precision in JSONL.
    python3 -c "print(round(float('$val'), 2))"
  fi
}

# Run idris example with `--epochs N --seed S`, capture stdout, return
# PERF_MS_PER_EP value or "crashed"/"missing".
run_idris() {
  local n="$1"
  local bin rc stdout_path errlog
  bin=$(binary_for_target)
  stdout_path=$(mktemp "${TMPDIR:-/tmp}/perf-baseline-out.XXXXXX")
  errlog=$(mktemp "${TMPDIR:-/tmp}/perf-baseline-err.XXXXXX")
  rc=0
  "$bin" --epochs "$n" --seed "$SEED" >"$stdout_path" 2>"$errlog" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[CRASH] $bin (epochs=$n) exit=$rc" >&2
    tail -3 "$errlog" >&2
    rm -f "$stdout_path" "$errlog"
    echo "crashed"
    return 0
  fi
  rm -f "$errlog"
  extract_marker "$stdout_path"
  rm -f "$stdout_path"
}

# Same for pytorch ref.
run_pytorch() {
  local n="$1"
  local stdout_path rc
  stdout_path=$(mktemp "${TMPDIR:-/tmp}/perf-baseline-out.XXXXXX")
  rc=0
  ( cd packages/pytorch && uv run python -m "$REF_MOD" --epochs "$n" --seed "$SEED" ) \
    >"$stdout_path" 2>/dev/null || rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$stdout_path"
    echo "crashed"
    return 0
  fi
  extract_marker "$stdout_path"
  rm -f "$stdout_path"
}

# Single-point in-script timing. Run once at N_LONG, parse the marker.
measure_idris() {
  build_idris_binary
  run_idris "$N_LONG"
}

measure_pytorch() {
  run_pytorch "$N_LONG"
}

# Capture the active commit (with +dirty marker), mirroring perf-run.sh.
# perf-log.jsonl is excluded from the dirty check — the perf scripts
# append to it mid-run, and that's not a code change worth flagging.
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [ -n "$(git status --porcelain -- ':!docs/develop/perf-log.jsonl' 2>/dev/null)" ]; then
  COMMIT="${COMMIT}+dirty"
fi

echo "[perf-baseline] $EXAMPLE_KEY [$BACKEND]: idris N=$N_LONG..." >&2
IDRIS_MS=$(measure_idris)
echo "[perf-baseline] $EXAMPLE_KEY: pytorch N=$N_LONG..." >&2
PY_MS=$(measure_pytorch)
if [ "$IDRIS_MS" = "crashed" ] || [ "$IDRIS_MS" = "missing" ]; then
  RATIO="N/A"
elif [ "$PY_MS" = "crashed" ] || [ "$PY_MS" = "missing" ]; then
  RATIO="N/A"
else
  RATIO=$(python3 -c "print(round(${IDRIS_MS} / ${PY_MS}, 2) if ${PY_MS} > 0 else 'inf')")
fi

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
idris_raw = os.environ["IDRIS_MS"]
py_raw = os.environ["PY_MS"]
idris_crashed = idris_raw in ("crashed", "missing")
py_crashed = py_raw in ("crashed", "missing")
row = {
    "ts": os.environ["ISO_TS"],
    "date": os.environ["DATE"],
    "kind": "baseline",
    "methodology": "in_script_marker",
    "example": os.environ["EXAMPLE_KEY"],
    "backend": os.environ["BACKEND"],
    "device": os.environ["DEVICE"],
    "commit": os.environ["COMMIT"],
    "idris_ms_per_epoch": None if idris_crashed else num(idris_raw),
    "pytorch_ms_per_epoch": None if py_crashed else num(py_raw),
    "ratio": None if (idris_crashed or py_crashed) else num(os.environ["RATIO"]),
    "n_long": int(os.environ["N_LONG"]),
    "seed": int(os.environ["SEED"]),
}
notes = []
if idris_raw == "crashed":
    notes.append("idris binary aborted during timed run")
elif idris_raw == "missing":
    notes.append("idris stdout had no PERF_MS_PER_EP marker")
if py_raw == "crashed":
    notes.append("pytorch ref aborted during timed run")
elif py_raw == "missing":
    notes.append("pytorch stdout had no PERF_MS_PER_EP marker")
if notes:
    row["notes"] = "; ".join(notes)
print(json.dumps(row))
PY

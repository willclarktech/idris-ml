#!/usr/bin/env bash
# scripts/perf-sweep.sh — sweep ms/epoch across (example × cell) cells.
#
# For each example: run the PyTorch reference ONCE (cached); then run
# Idris on each requested cell. One JSONL entry per (example, cell) is
# appended to docs/develop/perf-log.jsonl, plus a tabular summary to
# stdout.
#
# A "cell" is a (backend, device) pair. The supported cells are:
#   tape       — CPU, tape backend
#   torch      — CPU, libtorch backend
#   mlx-cpu    — mlx backend, MLX_DEVICE=cpu (default)
#   mlx-gpu    — mlx backend, MLX_DEVICE=gpu (Metal)
#
# Usage:
#   scripts/perf-sweep.sh [--examples a,b,...] [--cells tape,torch,mlx-cpu,mlx-gpu]
#                         [--seed N]
#
# Differences vs running scripts/perf-baseline.sh per cell:
#  - One PyTorch reference run per example (not per cell). On a 6-
#    example × 4-cell sweep that's 18 saved ref runs.
#  - mlx-gpu is a first-class cell (perf-baseline.sh has no flag for it).
#
# Timing methodology: per cell, build the example binary ONCE via
# `make example-<name>` (which links the right backend dylib), then
# run it at N_LONG and grep `PERF_MS_PER_EP=<float>` from stdout.
# Each side (Idris + PyTorch ref) emits the marker after the timed
# training loop, before eval — so the measurement is the per-epoch
# wall *inside* the training process, with no startup / build / eval
# variance folded in.
#
# Previous methodology was two-point wall-clock subtraction
# (`(wall(N_long) - wall(N_short)) / (N_long - N_short)`); see
# `docs/develop/perf-changes.md` 2026-05-19 entry for why it broke on
# short-converging RL refs.
set -euo pipefail

EXAMPLES_CSV="rnn,lstm,gru,transformer,ntm-copy,ntm-recall"
CELLS_CSV="tape,torch,mlx-cpu,mlx-gpu"
SEED=42

while [ $# -gt 0 ]; do
  case "$1" in
    --examples) EXAMPLES_CSV="$2"; shift 2 ;;
    --cells)    CELLS_CSV="$2"; shift 2 ;;
    --seed)     SEED="$2"; shift 2 ;;
    -h|--help)  sed -n '/^# scripts/,/^set -e/p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# example -> "make-target args-var pytorch-module N_SHORT N_LONG"
spec_for() {
  case "$1" in
    supervised)        echo "example-supervised SUPERVISED_ARGS torch_ref.scripts.supervised 50 200" ;;
    rnn)               echo "example-rnn RNN_ARGS torch_ref.scripts.rnn 50 200" ;;
    lstm)              echo "example-lstm LSTM_ARGS torch_ref.scripts.lstm 50 200" ;;
    gru)               echo "example-gru GRU_ARGS torch_ref.scripts.gru 50 200" ;;
    transformer)       echo "example-transformer TRANSFORMER_ARGS torch_ref.scripts.transformer 50 200" ;;
    ntm-copy)          echo "example-ntm-copy NTM_COPY_ARGS torch_ref.scripts.ntm_copy 10 40" ;;
    ntm-recall)        echo "example-ntm-associative-recall NTM_RECALL_ARGS torch_ref.scripts.ntm_recall 10 40" ;;
    dnc-copy)          echo "example-dnc-copy DNC_COPY_ARGS torch_ref.scripts.dnc_copy 20 80" ;;
    dnc-recall)        echo "example-dnc-recall DNC_RECALL_ARGS torch_ref.scripts.dnc_recall 10 40" ;;
    reinforce)         echo "example-reinforce REINFORCE_ARGS torch_ref.scripts.reinforce 50 200" ;;
    dqn)               echo "example-dqn DQN_ARGS torch_ref.scripts.dqn 20 80" ;;
    mountain-car)      echo "example-mountain-car MOUNTAIN_CAR_ARGS torch_ref.scripts.mountain_car 20 80" ;;
    mountain-car-cont) echo "example-mountain-car-cont MOUNTAIN_CAR_CONT_ARGS torch_ref.scripts.mountain_car_cont 1100 2000" ;;
    a2c)               echo "example-a2c A2C_ARGS torch_ref.scripts.a2c 50 200" ;;
    ppo)               echo "example-ppo PPO_ARGS torch_ref.scripts.ppo 10 40" ;;
    sac)               echo "example-sac SAC_ARGS torch_ref.scripts.sac 1100 2000" ;;
    *) return 1 ;;
  esac
}

# cell -> "BACKEND DEVICE"
cell_to_backend_device() {
  case "$1" in
    tape)    echo "tape cpu" ;;
    torch)   echo "torch cpu" ;;
    mlx-cpu) echo "mlx cpu" ;;
    mlx-gpu) echo "mlx gpu" ;;
    *) return 1 ;;
  esac
}

now_ms() { python3 -c 'import time; print(int(time.time_ns()/1_000_000))'; }

# Extract `PERF_MS_PER_EP=<float>` from a captured stdout file.
# Returns the value as text, or "missing" if no marker was emitted.
extract_marker() {
  local stdout_path="$1"
  local val
  val=$(grep -E '^PERF_MS_PER_EP=' "$stdout_path" | tail -1 | sed 's/^PERF_MS_PER_EP=//')
  if [ -z "$val" ]; then
    echo "missing"
  else
    python3 -c "print(round(float('$val'), 2))"
  fi
}

# Build the example binary for a (backend, device) cell ONCE, leaving
# ./build/exec/<name> and ./build/exec/<name>_app/libidrisml.dylib in
# place for direct invocation. The make-driven build + dylib copy is
# the bulk of per-call wallclock on tiny examples (rnn/gru tape are
# sub-ms/epoch real, vs ~100-500ms of make overhead per make invocation),
# so timing make invocations directly produced negative ms/ep on the
# L60 sweep (2026-05-18). Calling the binary directly removes that
# overhead from the timing path.
#
# The example-* make recipes don't have a "build only" mode — they
# always run the binary at the end. We pass `<VAR>=--epochs 1 --seed N`
# so the post-build run is cheap and effectively a warmup-we-discard.
build_idris_binary() {
  local target="$1" var="$2" backend="$3" device="$4"
  if [ "$backend" = "mlx" ]; then
    MLX_DEVICE="$device" BACKEND="$backend" make --no-print-directory "$target" \
      "$var=--epochs 1 --seed $SEED" >/dev/null 2>&1 || true
  else
    BACKEND="$backend" make --no-print-directory "$target" \
      "$var=--epochs 1 --seed $SEED" >/dev/null 2>&1 || true
  fi
}

# Derive binary path from make target: example-rnn -> ./build/exec/rnn.
binary_for_target() {
  local target="$1"
  echo "./build/exec/${target#example-}"
}

# Run one idris invocation. Captures stdout, parses PERF_MS_PER_EP
# marker. Returns the float as text, "crashed" on non-zero exit, or
# "missing" if the binary completed but didn't emit the marker.
run_idris_once() {
  local target="$1" _var="$2" n="$3" backend="$4" device="$5"
  local bin rc stdout_path errlog
  bin=$(binary_for_target "$target")
  stdout_path=$(mktemp "${TMPDIR:-/tmp}/perf-sweep-out.XXXXXX")
  errlog=$(mktemp "${TMPDIR:-/tmp}/perf-sweep-err.XXXXXX")
  rc=0
  if [ "$backend" = "mlx" ]; then
    MLX_DEVICE="$device" "$bin" --epochs "$n" --seed "$SEED" >"$stdout_path" 2>"$errlog" || rc=$?
  else
    "$bin" --epochs "$n" --seed "$SEED" >"$stdout_path" 2>"$errlog" || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[CRASH] $bin (backend=$backend device=$device epochs=$n) exit=$rc" >&2
    tail -3 "$errlog" >&2
    rm -f "$stdout_path" "$errlog"
    echo "crashed"
    return 0
  fi
  rm -f "$errlog"
  extract_marker "$stdout_path"
  rm -f "$stdout_path"
}

run_pytorch_once() {
  local mod="$1" n="$2"
  local stdout_path rc
  stdout_path=$(mktemp "${TMPDIR:-/tmp}/perf-sweep-out.XXXXXX")
  rc=0
  ( cd packages/pytorch && uv run python -m "$mod" --epochs "$n" --seed "$SEED" ) \
    >"$stdout_path" 2>/dev/null || rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$stdout_path"
    echo "crashed"
    return 0
  fi
  extract_marker "$stdout_path"
  rm -f "$stdout_path"
}

# Single-point in-script timing. Build binary once for the cell
# (relinks backend dylib), then run at N_LONG and parse the marker
# from stdout. No two-point subtraction.
measure_idris() {
  local target="$1" var="$2" n_long="$3" backend="$4" device="$5"
  build_idris_binary "$target" "$var" "$backend" "$device"
  run_idris_once "$target" "$var" "$n_long" "$backend" "$device"
}

measure_pytorch() {
  local mod="$1" n_long="$2"
  run_pytorch_once "$mod" "$n_long"
}

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
# Exclude perf-log.jsonl — the sweep itself appends to it mid-run,
# and that churn is not a code change worth flagging.
if [ -n "$(git status --porcelain -- ':!docs/develop/perf-log.jsonl' 2>/dev/null)" ]; then COMMIT="${COMMIT}+dirty"; fi
LOG_PATH="docs/develop/perf-log.jsonl"
[ -e "$LOG_PATH" ] || : > "$LOG_PATH"

write_jsonl_row() {
  local example="$1" backend="$2" device="$3" idris_ms="$4" py_ms="$5" ratio="$6" n_long="$7"
  local ts date
  ts=$(python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')
  date=$(date +%Y-%m-%d)
  ISO_TS="$ts" DATE="$date" EXAMPLE="$example" BACKEND="$backend" DEVICE="$device" \
    COMMIT="$COMMIT" IDRIS_MS="$idris_ms" PY_MS="$py_ms" RATIO="$ratio" \
    N_LONG="$n_long" SEED="$SEED" python3 <<'PY' >> "$LOG_PATH"
import json, os
def num(s):
    try: return float(s)
    except (ValueError, TypeError): return None
idris_raw = os.environ["IDRIS_MS"]
py_raw = os.environ["PY_MS"]
idris_crashed = idris_raw in ("crashed", "missing")
py_crashed = py_raw in ("crashed", "missing")
row = {
    "ts": os.environ["ISO_TS"], "date": os.environ["DATE"],
    "kind": "baseline",
    "example": os.environ["EXAMPLE"], "backend": os.environ["BACKEND"],
    "device": os.environ["DEVICE"], "commit": os.environ["COMMIT"],
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
}

echo
printf '%-18s %-8s %-6s %12s %12s %8s\n' example backend device idris_ms py_ms ratio
printf '%-18s %-8s %-6s %12s %12s %8s\n' '------' '------' '----' '--------' '-----' '-----'

IFS=, read -r -a EXAMPLES <<<"$EXAMPLES_CSV"
IFS=, read -r -a CELLS <<<"$CELLS_CSV"

for example in "${EXAMPLES[@]}"; do
  spec=$(spec_for "$example") || { echo "unknown example: $example" >&2; exit 2; }
  read -r IDRIS_TGT IDRIS_VAR REF_MOD N_SHORT N_LONG <<<"$spec"

  echo "[$example] pytorch ref N=$N_LONG..." >&2
  PY_MS=$(measure_pytorch "$REF_MOD" "$N_LONG")

  for cell in "${CELLS[@]}"; do
    bd=$(cell_to_backend_device "$cell") || { echo "unknown cell: $cell" >&2; exit 2; }
    read -r BACKEND DEVICE <<<"$bd"
    echo "[$example/$cell] idris N=$N_LONG..." >&2
    IDRIS_MS=$(measure_idris "$IDRIS_TGT" "$IDRIS_VAR" "$N_LONG" "$BACKEND" "$DEVICE")
    if [ "$IDRIS_MS" = "crashed" ] || [ "$IDRIS_MS" = "missing" ]; then
      RATIO="N/A"
    elif [ "$PY_MS" = "crashed" ] || [ "$PY_MS" = "missing" ]; then
      RATIO="N/A"
    else
      RATIO=$(python3 -c "print(round($IDRIS_MS / $PY_MS, 2) if $PY_MS > 0 else float('inf'))")
    fi
    write_jsonl_row "$example" "$BACKEND" "$DEVICE" "$IDRIS_MS" "$PY_MS" "$RATIO" "$N_LONG"
    printf '%-18s %-8s %-6s %12s %12s %8s\n' "$example" "$BACKEND" "$DEVICE" "$IDRIS_MS" "$PY_MS" "$RATIO"
  done
done

echo
echo "Sweep complete. JSONL entries appended to $LOG_PATH (kind=baseline)."

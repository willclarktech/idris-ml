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
    mountain-car-cont) echo "example-mountain-car-cont MOUNTAIN_CAR_CONT_ARGS torch_ref.scripts.mountain_car_cont 20 80" ;;
    a2c)               echo "example-a2c A2C_ARGS torch_ref.scripts.a2c 50 200" ;;
    ppo)               echo "example-ppo PPO_ARGS torch_ref.scripts.ppo 10 40" ;;
    sac)               echo "example-sac SAC_ARGS torch_ref.scripts.sac 20 80" ;;
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

# Run one idris invocation, return wall-clock ms.
run_idris_once() {
  local target="$1" var="$2" n="$3" backend="$4" device="$5"
  local t0 t1
  t0=$(now_ms)
  if [ "$backend" = "mlx" ]; then
    MLX_DEVICE="$device" BACKEND="$backend" make --no-print-directory "$target" \
      "$var=--epochs $n --seed $SEED" >/dev/null 2>&1
  else
    BACKEND="$backend" make --no-print-directory "$target" \
      "$var=--epochs $n --seed $SEED" >/dev/null 2>&1
  fi
  t1=$(now_ms)
  echo $((t1 - t0))
}

run_pytorch_once() {
  local mod="$1" n="$2"
  local t0 t1
  t0=$(now_ms)
  ( cd packages/pytorch && uv run python -m "$mod" --epochs "$n" --seed "$SEED" ) >/dev/null 2>&1
  t1=$(now_ms)
  echo $((t1 - t0))
}

# Two-point timing: (T_long - T_short) / (N_long - N_short). Warmup first.
two_point_idris() {
  local target="$1" var="$2" n_short="$3" n_long="$4" backend="$5" device="$6"
  run_idris_once "$target" "$var" "$n_short" "$backend" "$device" >/dev/null # warmup
  local t_short t_long
  t_short=$(run_idris_once "$target" "$var" "$n_short" "$backend" "$device")
  t_long=$(run_idris_once "$target" "$var" "$n_long" "$backend" "$device")
  python3 -c "print(round(($t_long - $t_short) / ($n_long - $n_short), 2))"
}

two_point_pytorch() {
  local mod="$1" n_short="$2" n_long="$3"
  run_pytorch_once "$mod" "$n_short" >/dev/null # warmup
  local t_short t_long
  t_short=$(run_pytorch_once "$mod" "$n_short")
  t_long=$(run_pytorch_once "$mod" "$n_long")
  python3 -c "print(round(($t_long - $t_short) / ($n_long - $n_short), 2))"
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
print(json.dumps({
    "ts": os.environ["ISO_TS"], "date": os.environ["DATE"],
    "kind": "baseline",
    "example": os.environ["EXAMPLE"], "backend": os.environ["BACKEND"],
    "device": os.environ["DEVICE"], "commit": os.environ["COMMIT"],
    "idris_ms_per_epoch": num(os.environ["IDRIS_MS"]),
    "pytorch_ms_per_epoch": num(os.environ["PY_MS"]),
    "ratio": num(os.environ["RATIO"]),
    "n_long": int(os.environ["N_LONG"]),
    "seed": int(os.environ["SEED"]),
}))
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

  echo "[$example] pytorch ref N=$N_SHORT then $N_LONG..." >&2
  PY_MS=$(two_point_pytorch "$REF_MOD" "$N_SHORT" "$N_LONG")

  for cell in "${CELLS[@]}"; do
    bd=$(cell_to_backend_device "$cell") || { echo "unknown cell: $cell" >&2; exit 2; }
    read -r BACKEND DEVICE <<<"$bd"
    echo "[$example/$cell] idris N=$N_SHORT then $N_LONG..." >&2
    IDRIS_MS=$(two_point_idris "$IDRIS_TGT" "$IDRIS_VAR" "$N_SHORT" "$N_LONG" "$BACKEND" "$DEVICE")
    RATIO=$(python3 -c "print(round($IDRIS_MS / $PY_MS, 2) if $PY_MS > 0 else float('inf'))")
    write_jsonl_row "$example" "$BACKEND" "$DEVICE" "$IDRIS_MS" "$PY_MS" "$RATIO" "$N_LONG"
    printf '%-18s %-8s %-6s %12s %12s %8s\n' "$example" "$BACKEND" "$DEVICE" "$IDRIS_MS" "$PY_MS" "$RATIO"
  done
done

echo
echo "Sweep complete. JSONL entries appended to $LOG_PATH (kind=baseline)."

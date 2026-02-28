#!/usr/bin/env bash
#
# Grid search over NTM hyperparameters.
# Usage: bash scripts/sweep.sh [--parallel N] [--skip-build] [--quick]
#
# Compiles once, then runs configs in parallel via xargs.
# Results are collected into results/sweep.csv sorted by test accuracy.
#
set -euo pipefail

PARALLEL=4
SKIP_BUILD=false
EPOCHS=6000
PATIENCE=500

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) PARALLEL="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --quick) EPOCHS=2000; shift ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

# Build once
if [[ "$SKIP_BUILD" == false ]]; then
  echo "Building NTM..."
  idris2 --source-dir src -p contrib -o ntm src/Example/Ntm.idr
fi

EXEC=./build/exec/ntm
if [[ ! -x "$EXEC" ]]; then
  echo "Error: $EXEC not found. Run without --skip-build." >&2
  exit 1
fi

mkdir -p results

# Grid values
LR_VALUES="0.003 0.001 0.0003"
MAX_NORM_VALUES="3.0 5.0 10.0"
SEED_VALUES="1 2 3 42"
BETA1_VALUES="0.9"

RESULTS_FILE="results/sweep.csv"
echo "lr,maxNorm,beta1,beta2,epochs,patience,epochsDone,seed,H,trainAcc,testAcc" > "$RESULTS_FILE"

# Generate all configs
CONFIGS=""
for lr in $LR_VALUES; do
  for maxNorm in $MAX_NORM_VALUES; do
    for beta1 in $BETA1_VALUES; do
      for seed in $SEED_VALUES; do
        CONFIGS="${CONFIGS}${lr} ${maxNorm} ${beta1} ${seed}\n"
      done
    done
  done
done

TOTAL=$(echo -e "$CONFIGS" | grep -c '[^ ]')
echo "Running $TOTAL configs with $PARALLEL parallel jobs (epochs=$EPOCHS, patience=$PATIENCE)..."
echo ""

TMPDIR_SWEEP=$(mktemp -d)
trap "rm -rf $TMPDIR_SWEEP" EXIT

# Run each config and extract RESULT line
run_one() {
  local lr=$1 maxNorm=$2 beta1=$3 seed=$4
  local tag="lr=${lr}_norm=${maxNorm}_b1=${beta1}_seed=${seed}"
  local outfile="${TMPDIR_SWEEP}/${tag}.out"

  "$EXEC" --lr "$lr" --max-norm "$maxNorm" --beta1 "$beta1" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$seed" \
    > "$outfile" 2>&1

  local result
  result=$(grep "^RESULT" "$outfile" || echo "")
  if [[ -n "$result" ]]; then
    # RESULT\tlr\tmaxNorm\tbeta1\tbeta2\tepochs\tpatience\tepochsDone\tseed\tH\ttrainAcc\ttestAcc
    echo "$result" | awk -F'\t' '{print $2","$3","$4","$5","$6","$7","$8","$9","$10","$11","$12}'
  else
    echo "$lr,$maxNorm,$beta1,0.999,$EPOCHS,$PATIENCE,0,$seed,?,-1,-1"
  fi
}
export -f run_one
export EXEC TMPDIR_SWEEP EPOCHS PATIENCE

echo -e "$CONFIGS" | grep '[^ ]' | \
  xargs -P "$PARALLEL" -L 1 bash -c 'run_one $@' _ | \
  tee -a "$RESULTS_FILE"

echo ""
echo "=== Results sorted by test accuracy ==="
echo ""
(head -1 "$RESULTS_FILE"; tail -n +2 "$RESULTS_FILE" | sort -t, -k11 -rn) | column -t -s,
echo ""
echo "Full results: $RESULTS_FILE"

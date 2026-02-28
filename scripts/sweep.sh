#!/usr/bin/env bash
#
# Grid search over NTM hyperparameters.
# Usage: bash scripts/sweep.sh [--parallel N] [--skip-build]
#
# Compiles once, then runs configs in parallel via xargs.
# Results are collected into results/sweep.csv sorted by test accuracy.
#
set -euo pipefail

PARALLEL=4
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) PARALLEL="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
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
LR1_VALUES="0.001 0.0005 0.0002"
MAX_NORM_VALUES="1.0 5.0 10.0"
SEED_VALUES="1 2 3"

# LR2 is always lr1 * 0.3
lr2_from_lr1() { echo "$1 * 0.3" | bc -l; }

RESULTS_FILE="results/sweep.csv"
echo "lr1,lr2,maxNorm,epochs1,epochs2,seed,H,trainAcc,testAcc" > "$RESULTS_FILE"

EPOCHS1=3000
EPOCHS2=3000

# Generate all configs
CONFIGS=""
for lr1 in $LR1_VALUES; do
  for maxNorm in $MAX_NORM_VALUES; do
    for seed in $SEED_VALUES; do
      lr2=$(lr2_from_lr1 "$lr1")
      CONFIGS="${CONFIGS}${lr1} ${lr2} ${maxNorm} ${EPOCHS1} ${EPOCHS2} ${seed}\n"
    done
  done
done

TOTAL=$(echo -e "$CONFIGS" | grep -c '[^ ]')
echo "Running $TOTAL configs with $PARALLEL parallel jobs..."
echo ""

TMPDIR_SWEEP=$(mktemp -d)
trap "rm -rf $TMPDIR_SWEEP" EXIT

# Run each config and extract RESULT line
run_one() {
  local lr1=$1 lr2=$2 maxNorm=$3 epochs1=$4 epochs2=$5 seed=$6
  local tag="lr1=${lr1}_norm=${maxNorm}_seed=${seed}"
  local outfile="${TMPDIR_SWEEP}/${tag}.out"

  "$EXEC" --lr1 "$lr1" --lr2 "$lr2" --max-norm "$maxNorm" \
    --epochs1 "$epochs1" --epochs2 "$epochs2" --seed "$seed" \
    > "$outfile" 2>&1

  local result
  result=$(grep "^RESULT" "$outfile" || echo "")
  if [[ -n "$result" ]]; then
    # RESULT\tlr1\tlr2\tmaxNorm\tepochs1\tepochs2\tseed\tH\ttrainAcc\ttestAcc
    echo "$result" | awk -F'\t' '{print $2","$3","$4","$5","$6","$7","$8","$9","$10}'
  else
    echo "$lr1,$lr2,$maxNorm,$epochs1,$epochs2,$seed,?,-1,-1"
  fi
}
export -f run_one
export EXEC TMPDIR_SWEEP

echo -e "$CONFIGS" | grep '[^ ]' | \
  xargs -P "$PARALLEL" -L 1 bash -c 'run_one $@' _ | \
  tee -a "$RESULTS_FILE"

echo ""
echo "=== Results sorted by test accuracy ==="
echo ""
(head -1 "$RESULTS_FILE"; tail -n +2 "$RESULTS_FILE" | sort -t, -k9 -rn) | column -t -s,
echo ""
echo "Full results: $RESULTS_FILE"

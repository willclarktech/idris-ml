#!/usr/bin/env bash
#
# Grid search over hyperparameters.
# Usage: bash scripts/sweep.sh [--task copy|recall|lstm] [--parallel N] [--skip-build] [--quick]
#
# Compiles once, then runs configs in parallel via xargs.
# Results are collected into results/sweep-{task}.csv sorted by test accuracy.
#
set -euo pipefail

PARALLEL=4
SKIP_BUILD=false
EPOCHS=6000
PATIENCE=500
TASK=copy

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) PARALLEL="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --quick) EPOCHS=2000; shift ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

case "$TASK" in
  copy)
    SRC_FILE=src/Example/NtmCopy.idr
    EXEC_NAME=ntm-copy
    ;;
  recall)
    SRC_FILE=src/Example/NtmAssociativeRecall.idr
    EXEC_NAME=ntm-associative-recall
    ;;
  lstm)
    SRC_FILE=src/Example/Lstm.idr
    EXEC_NAME=lstm
    ;;
  *)
    echo "Unknown task: $TASK (expected copy, recall, or lstm)" >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")/.."

# Build once
if [[ "$SKIP_BUILD" == false ]]; then
  echo "Building $EXEC_NAME..."
  idris2 --source-dir src -p contrib -o "$EXEC_NAME" "$SRC_FILE"
fi

EXEC="./build/exec/$EXEC_NAME"
if [[ ! -x "$EXEC" ]]; then
  echo "Error: $EXEC not found. Run without --skip-build." >&2
  exit 1
fi

mkdir -p results

# Grid values
RESULTS_FILE="results/sweep-${TASK}.csv"

if [[ "$TASK" == "lstm" ]]; then
  LR_VALUES="0.1 0.03 0.01 0.003"
  SEED_VALUES="1 2 3 42"

  echo "lr,epochs,patience,epochsDone,seed,loss" > "$RESULTS_FILE"

  CONFIGS=""
  for lr in $LR_VALUES; do
    for seed in $SEED_VALUES; do
      CONFIGS="${CONFIGS}${lr} ${seed}\n"
    done
  done
else
  LR_VALUES="0.003 0.001 0.0003"
  MAX_NORM_VALUES="3.0 5.0 10.0"
  SEED_VALUES="1 2 3 42"
  BETA1_VALUES="0.9"

  if [[ "$TASK" == "recall" ]]; then
    echo "lr,maxNorm,beta1,beta2,epochs,patience,epochsDone,seed,H,k2Acc,k3Acc,k5Acc" > "$RESULTS_FILE"
  else
    echo "lr,maxNorm,beta1,beta2,epochs,patience,epochsDone,seed,H,trainAcc,testAcc" > "$RESULTS_FILE"
  fi

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
fi

TOTAL=$(echo -e "$CONFIGS" | grep -c '[^ ]')
echo "Running $TOTAL configs with $PARALLEL parallel jobs (epochs=$EPOCHS, patience=$PATIENCE)..."
echo ""

TMPDIR_SWEEP=$(mktemp -d)
trap "rm -rf $TMPDIR_SWEEP" EXIT

# Run each config and extract RESULT line
run_one() {
  if [[ "$TASK" == "lstm" ]]; then
    local lr=$1 seed=$2
    local tag="lr=${lr}_seed=${seed}"
    local outfile="${TMPDIR_SWEEP}/${tag}.out"

    "$EXEC" --lr "$lr" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$seed" \
      > "$outfile" 2>&1

    local result
    result=$(grep "^RESULT" "$outfile" || echo "")
    if [[ -n "$result" ]]; then
      echo "$result" | awk -F'\t' '{s=$2; for(i=3;i<=NF;i++) s=s","$i; print s}'
    else
      echo "$lr,$EPOCHS,$PATIENCE,0,$seed,999"
    fi
  else
    local lr=$1 maxNorm=$2 beta1=$3 seed=$4
    local tag="lr=${lr}_norm=${maxNorm}_b1=${beta1}_seed=${seed}"
    local outfile="${TMPDIR_SWEEP}/${tag}.out"

    "$EXEC" --lr "$lr" --max-norm "$maxNorm" --beta1 "$beta1" \
      --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$seed" \
      > "$outfile" 2>&1

    local result
    result=$(grep "^RESULT" "$outfile" || echo "")
    if [[ -n "$result" ]]; then
      echo "$result" | awk -F'\t' '{s=$2; for(i=3;i<=NF;i++) s=s","$i; print s}'
    else
      if [[ "$TASK" == "recall" ]]; then
        echo "$lr,$maxNorm,$beta1,0.999,$EPOCHS,$PATIENCE,0,$seed,?,-1,-1,-1"
      else
        echo "$lr,$maxNorm,$beta1,0.999,$EPOCHS,$PATIENCE,0,$seed,?,-1,-1"
      fi
    fi
  fi
}
export -f run_one
export EXEC TMPDIR_SWEEP EPOCHS PATIENCE TASK

echo -e "$CONFIGS" | grep '[^ ]' | \
  xargs -P "$PARALLEL" -L 1 bash -c 'run_one $@' _ | \
  tee -a "$RESULTS_FILE"

echo ""
if [[ "$TASK" == "lstm" ]]; then
  echo "=== Results sorted by loss (ascending) ==="
  echo ""
  SORT_COL=6
  SORT_ORDER="n"
else
  echo "=== Results sorted by test accuracy ==="
  echo ""
  if [[ "$TASK" == "recall" ]]; then
    SORT_COL=12
  else
    SORT_COL=11
  fi
  SORT_ORDER="rn"
fi
(head -1 "$RESULTS_FILE"; tail -n +2 "$RESULTS_FILE" | sort -t, -k${SORT_COL} -${SORT_ORDER}) | column -t -s,
echo ""
echo "Full results: $RESULTS_FILE"

#!/usr/bin/env bash
#
# Generic hyperparameter grid / random search.
#
# Reads a sweep spec from a JSON file and runs Cartesian-product (or
# randomly-sampled) configs against the named example, capturing each
# run's RESULT line into a CSV.
#
# Usage:
#   bash scripts/sweep.sh --grid <spec.json> [--parallel N]
#                          [--epochs N] [--patience N]
#                          [--random N] [--skip-build]
#
# Convenience entry points (load scripts/sweeps/<task>.json):
#   bash scripts/sweep.sh --task copy [...]
#   bash scripts/sweep.sh --task recall [...]
#   bash scripts/sweep.sh --task lstm [...]
#
# Spec format (JSON):
#   {
#     "name": "ntm-copy",
#     "src":  "packages/idris-ml-examples/src/Example/NtmCopy.idr",
#     "exec": "ntm-copy",
#     "fixed_flags": ["--alpha", "0.95"],
#     "grid": {
#       "--lr":    [0.0001, 0.0003, 0.001],
#       "--batch": [4, 16],
#       "--seed":  [1, 2, 42]
#     }
#   }
#
# Behavior:
#   * Cartesian product of grid values, optionally subsampled with
#     `--random N` (uniform without replacement, deterministic if seeded
#     externally).
#   * `--epochs` and `--patience` are global flags (passed to every run);
#     leave the spec's grid free for the actual hyperparameters of
#     interest.
#   * RESULT lines are converted CSV-style: header = first RESULT's keys,
#     rows = each run's values. Configs that crashed/timed out emit a row
#     with the grid values + empty RESULT cells, so failures don't drop.
set -euo pipefail

PARALLEL=4
SKIP_BUILD=false
EPOCHS=6000
PATIENCE=500
RANDOM_N=""
GRID=""
TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)    PARALLEL="$2"; shift 2 ;;
    --skip-build)  SKIP_BUILD=true; shift ;;
    --quick)       EPOCHS=2000; shift ;;
    --epochs)      EPOCHS="$2"; shift 2 ;;
    --patience)    PATIENCE="$2"; shift 2 ;;
    --random)      RANDOM_N="$2"; shift 2 ;;
    --grid)        GRID="$2"; shift 2 ;;
    --task)        TASK="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

# Mirror Makefile's idris2 package-path setup so this script works
# without `make` being involved. IDRIS2_LOCAL is exported by the
# Makefile after the multi-build-key refactor (build/<BUILD_KEY>/
# idris2-prefix); fall back to the legacy `.idris2` at repo root
# for users invoking the script outside `make`.
IDRIS2_LOCAL="${IDRIS2_LOCAL:-$(pwd)/.idris2}"
SYS_IDRIS2_PREFIX="$(idris2 --paths 2>/dev/null | sed -n 's/.*Installation Prefix.*"\([^"]*\)".*/\1/p' || true)"
if [[ -z "$SYS_IDRIS2_PREFIX" ]]; then
  export IDRIS2_PACKAGE_PATH="$IDRIS2_LOCAL/idris2-0.8.0"
else
  export IDRIS2_PACKAGE_PATH="$IDRIS2_LOCAL/idris2-0.8.0:$SYS_IDRIS2_PREFIX/idris2-0.8.0"
fi

# Resolve --task convenience flag to the standard JSON specs.
if [[ -n "$TASK" && -z "$GRID" ]]; then
  GRID="scripts/sweeps/${TASK}.json"
fi

if [[ -z "$GRID" || ! -f "$GRID" ]]; then
  echo "Error: must pass --grid <spec.json> or --task {copy,recall,lstm}" >&2
  echo "       grid=$GRID" >&2
  exit 1
fi

NAME=$(jq -r '.name' "$GRID")
SRC=$(jq -r '.src' "$GRID")
EXEC_NAME=$(jq -r '.exec' "$GRID")
FIXED_FLAGS=$(jq -r '.fixed_flags // [] | join(" ")' "$GRID")

# Build once.
EXEC="./build/exec/$EXEC_NAME"
if [[ "$SKIP_BUILD" == false ]]; then
  echo "Building $EXEC_NAME from $SRC..."
  idris2 --source-dir packages/idris-ml-examples/src \
         -p contrib -p idris-ml -p idris-gym -p idris-ml-examples \
         -o "$EXEC_NAME" "$SRC"
fi
if [[ ! -x "$EXEC" ]]; then
  echo "Error: $EXEC not found. Run without --skip-build." >&2
  exit 1
fi

# Find libidrisml dylib (name varies by backend) and stage it next to
# the executable so chez can dlopen it.
DYLIB=$(ls build/libidrisml.dylib build/libidrisml*.dylib 2>/dev/null | head -1)
if [[ -n "$DYLIB" && -d "build/exec/${EXEC_NAME}_app" ]]; then
  cp "$DYLIB" "build/exec/${EXEC_NAME}_app/"
fi

# ------------------------------------------------------------------
# Generate configs (Cartesian product, optionally random-sampled).
# Each line of $CONFIGS_FILE is one config: a tab-separated list of
# "<flag>=<value>" pairs.
# ------------------------------------------------------------------

mkdir -p results
RESULTS_FILE="results/sweep-${NAME}.csv"
TMPDIR_SWEEP=$(mktemp -d)
trap "rm -rf $TMPDIR_SWEEP" EXIT
CONFIGS_FILE="$TMPDIR_SWEEP/configs"

# Read grid keys + values into parallel arrays via jq.
mapfile -t GRID_KEYS < <(jq -r '.grid | keys[]' "$GRID")
declare -a GRID_VALUES
for k in "${GRID_KEYS[@]}"; do
  vs=$(jq -r --arg k "$k" '.grid[$k] | map(tostring) | join(",")' "$GRID")
  GRID_VALUES+=("$vs")
done

# Recursive Cartesian product (writes lines to $CONFIGS_FILE).
cartesian_product() {
  local depth=$1; shift
  local prefix=$1; shift
  if [[ $depth -ge ${#GRID_KEYS[@]} ]]; then
    echo -e "$prefix" >> "$CONFIGS_FILE"
    return
  fi
  local key="${GRID_KEYS[$depth]}"
  IFS=',' read -ra vals <<< "${GRID_VALUES[$depth]}"
  for v in "${vals[@]}"; do
    local sep=$'\t'
    if [[ -z "$prefix" ]]; then sep=""; fi
    cartesian_product $((depth + 1)) "${prefix}${sep}${key}=${v}"
  done
}
cartesian_product 0 ""

TOTAL=$(wc -l < "$CONFIGS_FILE" | tr -d ' ')

# Optional: random subsample.
if [[ -n "$RANDOM_N" && "$RANDOM_N" -lt "$TOTAL" ]]; then
  shuf -n "$RANDOM_N" "$CONFIGS_FILE" > "$TMPDIR_SWEEP/sampled" \
    && mv "$TMPDIR_SWEEP/sampled" "$CONFIGS_FILE"
  TOTAL="$RANDOM_N"
fi

echo "Running $TOTAL configs with $PARALLEL parallel jobs (epochs=$EPOCHS, patience=$PATIENCE)..."
echo ""

# ------------------------------------------------------------------
# Execute configs and collect RESULT lines.
# ------------------------------------------------------------------

run_one() {
  local config_line="$1"
  # Build the flag-list: each "key=value" → "key value"
  local flags=""
  local tag=""
  IFS=$'\t' read -ra parts <<< "$config_line"
  for p in "${parts[@]}"; do
    local k="${p%%=*}"
    local v="${p#*=}"
    flags+="$k $v "
    tag+="${k#--}=${v}_"
  done
  tag="${tag%_}"
  local outfile="${TMPDIR_SWEEP}/${tag}.out"

  # shellcheck disable=SC2086
  "$EXEC" $flags --epochs "$EPOCHS" --patience "$PATIENCE" $FIXED_FLAGS \
    > "$outfile" 2>&1 || true

  local result
  result=$(grep "^RESULT" "$outfile" | head -1 || true)
  echo -e "${config_line}\t${result}" >> "$TMPDIR_SWEEP/raw_results"
}
export -f run_one
export EXEC TMPDIR_SWEEP EPOCHS PATIENCE FIXED_FLAGS

# Run in parallel. macOS xargs lacks `-d`, so we feed null-terminated
# config lines (configs may contain spaces but won't contain NULs).
tr '\n' '\0' < "$CONFIGS_FILE" | \
  xargs -0 -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ "{}"

# ------------------------------------------------------------------
# Convert tab-separated raw results to CSV.
# Header: grid keys (without `--` prefix) + RESULT keys from the first
#         non-empty RESULT line.
# Rows:   one per config; missing RESULT cells are empty.
# ------------------------------------------------------------------

# Strip the `--` prefix from grid keys for column headers.
HEADER_GRID=""
for k in "${GRID_KEYS[@]}"; do
  HEADER_GRID+="${k#--},"
done

# Find a sample RESULT to extract its keys for the CSV header.
SAMPLE_RESULT=$(awk -F'\t' '/RESULT/{for(i=1;i<=NF;i++) if($i ~ /^[a-zA-Z_]+=/) printf "%s,", $i; exit}' "$TMPDIR_SWEEP/raw_results" || true)
RESULT_HEADER=$(echo "$SAMPLE_RESULT" | tr ',' '\n' | awk -F'=' '{print $1}' | tr '\n' ',' | sed 's/,$//')

echo "${HEADER_GRID%,},${RESULT_HEADER}" > "$RESULTS_FILE"

# Convert each raw line to CSV.
while IFS= read -r line; do
  # Split into config_part \t RESULT...
  local_config=$(echo "$line" | awk -F'\tRESULT' '{print $1}')
  local_result=$(echo "$line" | awk -F'\tRESULT' '{print $2}')

  # Config: extract values from k=v pairs separated by tabs
  cfg_csv=""
  IFS=$'\t' read -ra parts <<< "$local_config"
  for p in "${parts[@]}"; do
    cfg_csv+="${p#*=},"
  done
  cfg_csv="${cfg_csv%,}"

  # Result: each tab-separated k=v becomes a CSV cell holding v.
  res_csv=""
  if [[ -n "$local_result" ]]; then
    IFS=$'\t' read -ra rparts <<< "$local_result"
    for p in "${rparts[@]}"; do
      [[ -z "$p" ]] && continue
      res_csv+="${p#*=},"
    done
    res_csv="${res_csv%,}"
  fi

  echo "${cfg_csv},${res_csv}" >> "$RESULTS_FILE"
done < "$TMPDIR_SWEEP/raw_results"

echo ""
echo "Wrote $TOTAL configs to $RESULTS_FILE"
echo ""

# Pretty-print, sorted by the RIGHTMOST result column (typically
# accuracy or loss; the user can post-process if they want a different
# sort key).
column -t -s, "$RESULTS_FILE"

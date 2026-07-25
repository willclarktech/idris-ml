#!/usr/bin/env bash
# Reference-side convergence campaign: the PyTorch peer of
# scripts/test-convergence.sh. Runs each torch_ref.scripts module at its own
# defaults across SEEDS, checks the RESULT line against
# test-refs-convergence.expect, and appends one resumable row per
# (module, seed) to CONVERGENCE_REF_OUT.
#
# The TSV format and the pass-rate table are identical to the Idris campaign's,
# so the two tables sit side by side in reference-alignment.md and answer the
# same question at the same bars. `--report <tsv>` delegates to
# test-convergence.sh, which owns the tally.
#
# Individual misses are recorded data, not a gate failure — the point is the
# pass rate over >= 5 seeds (the multi-seed convergence policy), not a green
# tick. Exit is 0 whenever every run produced a checkable RESULT.
#
# Env interface:
#   MODULES              space-separated torch_ref.scripts module names
#   SEEDS                space-separated seeds (default: 42 1 2 3 4)
#   CONVERGENCE_TIMEOUT  per-run timeout in seconds (default 4h)
#   CONVERGENCE_EXPECT   expect file with thresholds
#   CONVERGENCE_REF_OUT  results TSV
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODULES=${MODULES:-}
SEEDS=${SEEDS:-"42 1 2 3 4"}
CONVERGENCE_TIMEOUT=${CONVERGENCE_TIMEOUT:-14400}
CONVERGENCE_EXPECT=${CONVERGENCE_EXPECT:-test-refs-convergence.expect}
CONVERGENCE_REF_OUT=${CONVERGENCE_REF_OUT:-docs/develop/convergence-campaign-ref.tsv}

if [ "${1:-}" = "--report" ]; then
	exec bash scripts/test-convergence.sh --report "${2:-$CONVERGENCE_REF_OUT}"
fi

if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $CONVERGENCE_TIMEOUT"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $CONVERGENCE_TIMEOUT"
else TIMEOUT_PREFIX=""; fi

echo "WARNING: full-convergence reference runs take several hours."
echo "         Press Ctrl-C in the next 5s to abort." && sleep 5

mkdir -p "$(dirname "$CONVERGENCE_REF_OUT")"; touch "$CONVERGENCE_REF_OUT"
echo "Campaign: $(echo "$MODULES" | wc -w | tr -d ' ') modules x [$SEEDS] seeds → $CONVERGENCE_REF_OUT"

fmt_elapsed() { # $1 seconds
	if [ "$1" -lt 60 ]; then echo "${1}s"
	elif [ "$1" -lt 3600 ]; then echo "$(($1/60))m$(($1%60))s"
	else echo "$(($1/3600))h$((($1%3600)/60))m"; fi
}

for m in $MODULES; do
	for s in $SEEDS; do
		if awk -F'\t' -v m="$m" -v s="$s" '$1==m && $2==s {f=1} END{exit !f}' "$CONVERGENCE_REF_OUT"; then
			echo "skip $m seed=$s (already recorded)"; continue
		fi
		echo "=== $m seed=$s ==="
		t_start=$(date +%s)
		output=$(cd packages/pytorch && $TIMEOUT_PREFIX uv run python -u -m "torch_ref.scripts.$m" --seed "$s" 2>&1); rc=$?
		elapsed_fmt=$(fmt_elapsed $(( $(date +%s) - t_start )))
		result_line=$(echo "$output" | grep '^RESULT' | head -1)

		status=""
		if [ $rc -eq 124 ]; then
			status="timeout"; echo "FAIL: $m timed out (>${CONVERGENCE_TIMEOUT}s) ($elapsed_fmt)"
			echo "$output" | tail -30 | sed 's/^/  | /'
		elif [ $rc -ne 0 ]; then
			status="crash"; echo "FAIL: $m crashed (rc=$rc) ($elapsed_fmt)"
			echo "$output" | tail -30 | sed 's/^/  | /'
		elif [ -z "$result_line" ]; then
			status="noresult"; echo "FAIL: $m -- no RESULT line ($elapsed_fmt)"
			echo "$output" | tail -30 | sed 's/^/  | /'
		elif scripts/check-result.sh "$m" "$result_line" "$CONVERGENCE_EXPECT"; then
			status="pass"; echo "  ($elapsed_fmt)"
		else
			status="fail"; echo "  ($elapsed_fmt)"
		fi

		printf '%s\t%s\t%s\t%s\t%s\n' "$m" "$s" "$status" "$elapsed_fmt" "${result_line#RESULT$'\t'}" \
			>> "$CONVERGENCE_REF_OUT"
	done
done

echo; echo "Campaign done. Pass-rate table:"
bash scripts/test-convergence.sh --report "$CONVERGENCE_REF_OUT"

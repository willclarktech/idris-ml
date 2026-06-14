#!/usr/bin/env bash
# Convergence runner: every example at full default epochs, tape backend,
# tight thresholds from test-examples-convergence.expect. Hours of wall
# time (NTM/DNC dominate). Intended for release validation, not CI.
# See docs/develop/testing.md.
#
# Two modes (same run → RESULT → check-result loop):
#
#   * GATE (default): single seed, text PASS/FAIL, exit 1 on any failure.
#     `make test-convergence`. SEEDS defaults to 42.
#
#   * CAMPAIGN: set CONVERGENCE_OUT to a TSV path → loops SEEDS (default
#     "42 1 2 3 4"), appends one resumable row per (example, seed), and
#     prints a markdown pass-rate table. A seed that misses its threshold
#     is recorded data, not a gate failure (the "multi-seed convergence
#     is required" alignment policy: report the pass rate over >= 5 seeds).
#     Resumable: an (example, seed) already in the TSV is skipped, so a
#     killed run continues. `make test-convergence-campaign`.
#     `scripts/test-convergence.sh --report <tsv>` prints the table from
#     an existing TSV without running anything.
#
# Invoked with the mandatory `+` recipe prefix (keeps the jobserver alive
# for the $MAKE sub-builds). Direct invocation works too — inputs default.
#
# Env interface:
#   MAKE                 make binary for sub-builds (default: make)
#   EXAMPLES             space-separated example-* targets
#   SEEDS                space-separated seeds (gate: 42; campaign: 42 1 2 3 4)
#   CONVERGENCE_TIMEOUT  per-run timeout in seconds (default 4h)
#   CONVERGENCE_EXPECT   expect file with convergence thresholds
#   CONVERGENCE_OUT      campaign results TSV (unset → gate mode)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAKE=${MAKE:-make}
EXAMPLES=${EXAMPLES:-}
CONVERGENCE_TIMEOUT=${CONVERGENCE_TIMEOUT:-14400}
CONVERGENCE_EXPECT=${CONVERGENCE_EXPECT:-test-examples-convergence.expect}
CONVERGENCE_OUT=${CONVERGENCE_OUT:-}

if [ -n "$CONVERGENCE_OUT" ]; then SEEDS=${SEEDS:-"42 1 2 3 4"}; else SEEDS=${SEEDS:-42}; fi

# ---- report mode: tally an existing TSV into a markdown pass-rate table --
report() {
	local out=$1
	[ -f "$out" ] || { echo "no results at $out" >&2; return 1; }
	echo "| Example | Pass rate | Seeds (✓ pass / ✗ fail) |"
	echo "|---------|-----------|--------------------------|"
	for e in $(awk -F'\t' '!seen[$1]++ {print $1}' "$out"); do  # distinct, first-seen order
		local rows total pass detail
		rows=$(awk -F'\t' -v e="$e" '$1==e' "$out")
		total=$(echo "$rows" | wc -l | tr -d ' ')
		pass=$(echo "$rows" | awk -F'\t' '$3=="pass"' | wc -l | tr -d ' ')
		detail=$(echo "$rows" | awk -F'\t' '{s=($3=="pass")?"✓":($3=="fail")?"✗":$3; printf "%s=%s ", $2, s}')
		printf '| %s | %s/%s | %s |\n' "$e" "$pass" "$total" "$detail"
	done
}

if [ "${1:-}" = "--report" ]; then
	report "${2:-$CONVERGENCE_OUT}"
	exit $?
fi

if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $CONVERGENCE_TIMEOUT"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $CONVERGENCE_TIMEOUT"
else TIMEOUT_PREFIX=""; fi

echo "WARNING: full-convergence runs take several hours on tape."
echo "         Press Ctrl-C in the next 5s to abort." && sleep 5

if [ -n "$CONVERGENCE_OUT" ]; then
	mkdir -p "$(dirname "$CONVERGENCE_OUT")"; touch "$CONVERGENCE_OUT"
	echo "Campaign: $(echo "$EXAMPLES" | wc -w | tr -d ' ') examples x [$SEEDS] seeds → $CONVERGENCE_OUT"
fi

fmt_elapsed() { # $1 seconds
	if [ "$1" -lt 60 ]; then echo "${1}s"
	elif [ "$1" -lt 3600 ]; then echo "$(($1/60))m$(($1%60))s"
	else echo "$(($1/3600))h$((($1%3600)/60))m"; fi
}

fail=0
for e in $EXAMPLES; do
	for s in $SEEDS; do
		if [ -n "$CONVERGENCE_OUT" ] && \
		   awk -F'\t' -v e="$e" -v s="$s" '$1==e && $2==s {f=1} END{exit !f}' "$CONVERGENCE_OUT"; then
			echo "skip $e seed=$s (already recorded)"; continue
		fi
		echo "=== $e seed=$s ==="
		t_start=$(date +%s)
		output=$($TIMEOUT_PREFIX $MAKE --no-print-directory BACKEND=tape "$e" SEED_FLAG="--seed $s" 2>&1); rc=$?
		elapsed_fmt=$(fmt_elapsed $(( $(date +%s) - t_start )))
		result_line=$(echo "$output" | grep '^RESULT' | head -1)

		status=""
		if [ $rc -eq 124 ]; then
			status="timeout"; echo "FAIL: $e timed out (>${CONVERGENCE_TIMEOUT}s) ($elapsed_fmt)"
			echo "$output" | tail -30 | sed 's/^/  | /'
		elif [ $rc -ne 0 ]; then
			status="crash"; echo "FAIL: $e crashed (rc=$rc) ($elapsed_fmt)"
			echo "$output" | tail -30 | sed 's/^/  | /'
		elif [ -z "$result_line" ]; then
			status="noresult"; echo "FAIL: $e -- no RESULT line ($elapsed_fmt)"
			echo "$output" | tail -30 | sed 's/^/  | /'
		elif scripts/check-result.sh "$e" "$result_line" "$CONVERGENCE_EXPECT"; then
			status="pass"; echo "  ($elapsed_fmt)"
		else
			status="fail"; echo "  ($elapsed_fmt)"
		fi

		if [ -n "$CONVERGENCE_OUT" ]; then
			printf '%s\t%s\t%s\t%s\t%s\n' "$e" "$s" "$status" "$elapsed_fmt" "${result_line#RESULT$'\t'}" >> "$CONVERGENCE_OUT"
		fi
		[ "$status" = "pass" ] || fail=1
	done
done

# Campaign mode: individual misses are data, not a gate failure — emit the
# table and exit 0. Gate mode: any failure fails the run.
if [ -n "$CONVERGENCE_OUT" ]; then
	echo; echo "Campaign done. Pass-rate table:"; report "$CONVERGENCE_OUT"
	exit 0
fi
if [ $fail -ne 0 ]; then echo "Some convergence runs FAILED"; exit 1; fi
echo "All convergence runs passed."

#!/usr/bin/env bash
# Full-convergence gate: every example at full default epochs, single
# seed=42, tape backend, tight thresholds from
# test-examples-convergence.expect. Hours of wall time (NTM/DNC dominate).
# Intended for release validation, not CI. See docs/develop/testing.md.
#
# Invoked by `make test-convergence` (mandatory `+` recipe prefix keeps
# the jobserver alive for the $MAKE sub-builds). Direct invocation works
# too — every input defaults below.
#
# Env interface (all passed by the Make recipe):
#   MAKE                 make binary for sub-builds (default: make)
#   EXAMPLES             space-separated example-* targets
#   CONVERGENCE_TIMEOUT  per-example timeout in seconds (default 4h)
#   CONVERGENCE_EXPECT   expect file with convergence thresholds

set -u

MAKE=${MAKE:-make}
EXAMPLES=${EXAMPLES:-}
CONVERGENCE_TIMEOUT=${CONVERGENCE_TIMEOUT:-14400}
CONVERGENCE_EXPECT=${CONVERGENCE_EXPECT:-test-examples-convergence.expect}

echo "WARNING: full-convergence runs take several hours on tape."
echo "         Press Ctrl-C in the next 5s to abort." && sleep 5

fail=0
if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $CONVERGENCE_TIMEOUT"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $CONVERGENCE_TIMEOUT"
else TIMEOUT_PREFIX=""; fi

for e in $EXAMPLES; do
	echo "=== $e ==="
	t_start=$(date +%s)
	output=$($TIMEOUT_PREFIX $MAKE --no-print-directory BACKEND=tape $e 2>&1); rc=$?
	t_end=$(date +%s); elapsed=$((t_end - t_start))
	if [ $elapsed -lt 60 ]; then elapsed_fmt="${elapsed}s"
	elif [ $elapsed -lt 3600 ]; then elapsed_fmt="$((elapsed/60))m$((elapsed%60))s"
	else elapsed_fmt="$((elapsed/3600))h$(((elapsed%3600)/60))m"; fi
	if [ $rc -ne 0 ]; then
		if [ $rc -eq 124 ]; then
			echo "FAIL: $e timed out (>${CONVERGENCE_TIMEOUT}s) ($elapsed_fmt)"
		else
			echo "FAIL: $e crashed (rc=$rc) ($elapsed_fmt)"
		fi
		echo "$output" | tail -30 | sed 's/^/  | /'
		fail=1; continue
	fi
	result_line=$(echo "$output" | grep '^RESULT' | head -1)
	if [ -z "$result_line" ]; then
		echo "FAIL: $e -- no RESULT line ($elapsed_fmt)"
		echo "$output" | tail -30 | sed 's/^/  | /'
		fail=1; continue
	fi
	scripts/check-result.sh "$e" "$result_line" "$CONVERGENCE_EXPECT" || fail=1
	echo "  ($elapsed_fmt)"
done

if [ $fail -ne 0 ]; then echo "Some convergence runs FAILED"; exit 1; fi
echo "All convergence runs passed."

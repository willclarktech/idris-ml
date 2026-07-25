#!/usr/bin/env bash
# scripts/check-convergence-expect-coverage.sh
#
# Gate: every example in the convergence campaign must have a threshold
# row in the convergence expect file.
#
# Why this exists: `scripts/check-result.sh` treats a MISSING expect row as
# "presence-only" and exits 0. That is the right default for the smoke lane
# (many examples have no meaningful scalar bar), but in the convergence
# campaign it means a listed example silently records `pass` without any
# assertion. `example-double-dqn` did exactly that — it landed 2026-06-19
# with only a crash-only smoke bar, its convergence row was deferred to
# "the campaign", and the campaign then auto-passed it at avg_return=85.0,
# below every example-dqn seed (101-162). Nothing detected the drift.
#
# The campaign list lives in mk/e2e.mk and the thresholds live in the
# expect file; they are edited independently, so they drift. This asks
# make for the list rather than re-deriving it (a hand-maintained copy
# would rot exactly like the thing it is checking).
#
# Usage: scripts/check-convergence-expect-coverage.sh [<expect-file> [<make-var>]]
#
# <make-var> names the make variable holding the campaign list, so the same
# gate covers the reference-side campaign (CONVERGENCE_REF_MODULES against
# test-refs-convergence.expect) — that list can drift from its thresholds the
# same way.
set -euo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )/.."

EXPECT="${1:-test-examples-convergence.expect}"
LIST_VAR="${2:-CONVERGENCE_CAMPAIGN_EXAMPLES}"

if [ ! -f "$EXPECT" ]; then
	echo "FAIL: expect file not found: $EXPECT" >&2
	exit 1
fi

# `print-%` is a prerequisite-free echo target (mk/config.mk), so this
# neither builds nor touches anything.
EXAMPLES=$( make -s --no-print-directory "print-$LIST_VAR" )

if [ -z "$EXAMPLES" ]; then
	echo "FAIL: $LIST_VAR is empty — did the variable move?" >&2
	exit 1
fi

missing=""
for e in $EXAMPLES; do
	# A threshold row is "<target> <key> <op> <value>" at line start.
	if ! grep -qE "^${e}[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+" "$EXPECT"; then
		missing="$missing $e"
	fi
done

if [ -n "$missing" ]; then
	echo "FAIL: convergence-campaign examples with no threshold row in $EXPECT:" >&2
	for e in $missing; do
		echo "  - $e" >&2
	done
	echo "" >&2
	echo "check-result.sh treats a missing row as presence-only (exit 0), so these" >&2
	echo "would record 'pass' in the campaign without asserting anything. Add a row" >&2
	echo "(anchored to the paired PyTorch reference's achieved rate) or drop the" >&2
	echo "entry from $LIST_VAR." >&2
	exit 1
fi

count=$( echo "$EXAMPLES" | wc -w | tr -d ' ' )
echo "check-convergence-expect-coverage: OK ($count entries in $LIST_VAR, all have thresholds)"

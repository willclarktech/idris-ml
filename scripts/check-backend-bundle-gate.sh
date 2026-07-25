#!/usr/bin/env bash
# Verifies the Backend-bundle availability gate: `Backend ex dt` must
# NOT resolve for an executor lacking a `Linked` instance. Inverts the
# idris2 exit code (success = compile failed) and asserts the error
# names the bundle, so unrelated regressions don't pass the gate.
#
# Run from the repo root via `make test-integration-typegate-backend-linked`
# (which provides IDRIS2_LOCAL after `make install`).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/src/Test/neg/BackendRequiresLinked.idr"
IDRIS_LOCAL="${IDRIS2_LOCAL:-$REPO_ROOT/.idris2}"

# Single compiler: pack's idris2 (matches the .ttc the build installed).
# IDRIS2 is exported by the Makefile; fall back to pack/PATH when run
# standalone. PACK_PKG_PATH adds the collection libs (elab-util, contrib,
# linear, ...) the local prefix alone doesn't carry.
IDRIS2="${IDRIS2:-$(pack app-path idris2 2>/dev/null || command -v idris2)}"
PACK_PKG_PATH="$(pack package-path 2>/dev/null || true)"

if [ ! -f "$NEG_FILE" ]; then
	echo "FAIL: negative test file missing at $NEG_FILE" >&2
	exit 1
fi

if [ ! -d "$IDRIS_LOCAL" ]; then
	echo "FAIL: $IDRIS_LOCAL not found — run 'make install' first (sets IDRIS2_LOCAL)" >&2
	exit 1
fi

cd "$(dirname "$NEG_FILE")"
OUTPUT="$(IDRIS2_PACKAGE_PATH="$IDRIS_LOCAL/idris2-0.8.0${PACK_PKG_PATH:+:$PACK_PKG_PATH}" \
					"$IDRIS2" --check "$(basename "$NEG_FILE")" -p idris-ml -p idris-random 2>&1 || true)"

# Success path: instance SEARCH failed naming the missing LEAF —
# Idris 0.8.0 walks through the blanket implementation and reports
# "Can't find an implementation for Linked FakeExecutor" (observed;
# better than the top-level Backend goal). Grepping the search-failure
# phrase specifically — NOT just a type name, which idris2 echoes in
# source-context lines even for unrelated errors.
if echo "$OUTPUT" | grep "Can't find an implementation for" | grep -q "Linked FakeExecutor"; then
	echo "PASS: bundle is unresolvable without Linked (search fails naming the leaf)"
	exit 0
fi

if echo "$OUTPUT" | grep -q "Undefined name Backend"; then
	echo "FAIL: Backend bundle module is missing entirely (Undefined" >&2
	echo "      name) — that's absence, not the Linked gate working." >&2
elif echo "$OUTPUT" | grep -q "BackendRequiresLinked:"; then
	echo "FAIL: negative test errored, but not as a Backend instance-" >&2
	echo "      search failure — gate may have regressed." >&2
else
	echo "FAIL: negative test compiled cleanly — the bundle no longer" >&2
	echo "      requires Linked; an unlinked backend could be spelled." >&2
fi
echo "--- idris2 output ---" >&2
echo "$OUTPUT" >&2
exit 1

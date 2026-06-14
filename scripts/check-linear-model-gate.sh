#!/usr/bin/env bash
# Verifies the linear-model gate: reusing a model handle after it has been
# consumed (by `evalL` / `freezeL`) must be a COMPILE-TIME linearity error —
# the typed defence against the "freeze/eval a model, then reuse the stale
# handle to train (silent no-op)" bug class.
#
# Two halves:
#   * NEG (Test/neg/ReuseAfterFreeze.idr) MUST NOT compile, and must fail
#     with a *linearity* error (so an unrelated regression doesn't pass the
#     gate). v0.8's message is "There are N uses of linear name …" or
#     "… is not accessible in this context".
#   * POS (Test/pos/SingleUseCompiles.idr) MUST compile — proving the neg
#     test fails for the right reason, not because the whole `ModuleL`/`evalL`
#     surface is broken.
#
# Run from the repo root via `make test-integration-typegate-linear-model`,
# after `make install` (so idris-ml + linear are in IDRIS2_PACKAGE_PATH).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/src/Test/neg/ReuseAfterFreeze.idr"
POS_FILE="$REPO_ROOT/packages/idris-ml/src/Test/pos/SingleUseCompiles.idr"
IDRIS_LOCAL="${IDRIS2_LOCAL:-$REPO_ROOT/.idris2}"

for f in "$NEG_FILE" "$POS_FILE"; do
	if [ ! -f "$f" ]; then
		echo "FAIL: gate test file missing at $f" >&2
		exit 1
	fi
done

if [ ! -d "$IDRIS_LOCAL" ]; then
	echo "FAIL: $IDRIS_LOCAL not found — run 'make install' first (sets IDRIS2_LOCAL)" >&2
	exit 1
fi

PKG_PATH="$IDRIS_LOCAL/idris2-0.8.0"

check() { # <file>  -> echoes idris2 output, returns idris2 exit code
	local f="$1"
	( cd "$(dirname "$f")" && \
		IDRIS2_PACKAGE_PATH="$PKG_PATH" \
			idris2 --check "$(basename "$f")" -p idris-ml -p linear -p contrib 2>&1 )
}

# ---- NEG: must fail, with a linearity error ----
NEG_OUT="$(check "$NEG_FILE")"
NEG_RC=$?

if [ "$NEG_RC" -eq 0 ]; then
	echo "FAIL: negative test compiled cleanly — the linear-model gate is" >&2
	echo "      broken. A consumed model handle can be reused; the 'stale" >&2
	echo "      model alias silently no-ops training' bug class is no longer" >&2
	echo "      caught." >&2
	echo "" >&2
	echo "--- idris2 output (neg) ---" >&2
	echo "$NEG_OUT" >&2
	exit 1
fi

if ! echo "$NEG_OUT" | grep -Eq "uses of linear name|not accessible in this context|linearly bounded"; then
	echo "FAIL: negative test errored, but not with a linearity error —" >&2
	echo "      the gate may have regressed to a different failure mode." >&2
	echo "" >&2
	echo "--- idris2 output (neg) ---" >&2
	echo "$NEG_OUT" >&2
	exit 1
fi

# ---- POS: must compile ----
POS_OUT="$(check "$POS_FILE")"
POS_RC=$?

if [ "$POS_RC" -ne 0 ]; then
	echo "FAIL: positive test did NOT compile — the neg test can't be" >&2
	echo "      trusted to fail for the right reason (the ModuleL/evalL" >&2
	echo "      surface itself may be broken)." >&2
	echo "" >&2
	echo "--- idris2 output (pos) ---" >&2
	echo "$POS_OUT" >&2
	exit 1
fi

echo "PASS: linear-model gate rejects reuse-after-consume (neg), accepts single-use (pos)"
exit 0

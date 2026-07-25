#!/usr/bin/env bash
# Verifies the Seq shape gate: a chain whose hidden dims don't line up must
# fail to compile with an error that NAMES BOTH DIMS — the `ChainFits`
# witness on `Nn.Seq`'s `(::)` turns the opaque pre-2026-07 failure
# (`Can't find an implementation for Module ?l`) into
# `Can't find an implementation for ChainFits 256 128`.
#
# Two halves:
#   * NEG (Test/neg/SeqShapeMismatch.idr) MUST NOT compile, and must fail
#     with a ChainFits error naming 256 and 128 (so an unrelated regression
#     doesn't pass the gate).
#   * POS (Test/pos/SeqChainCompiles.idr) MUST compile — proving the neg
#     test fails for the right reason, not because the Seq/ChainFits
#     surface is broken (it also routes an identity activation's dims
#     through two ChainFits ties).
#
# Run from the repo root via `make test-integration-typegate-seq-shape`,
# after `make install` (so idris-ml + linear are in IDRIS2_PACKAGE_PATH).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/src/Test/neg/SeqShapeMismatch.idr"
POS_FILE="$REPO_ROOT/packages/idris-ml/src/Test/pos/SeqChainCompiles.idr"
IDRIS_LOCAL="${IDRIS2_LOCAL:-$REPO_ROOT/.idris2}"

# Single compiler: pack's idris2 (matches the .ttc the build installed).
IDRIS2="${IDRIS2:-$(pack app-path idris2 2>/dev/null || command -v idris2)}"
PACK_PKG_PATH="$(pack package-path 2>/dev/null || true)"

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

PKG_PATH="$IDRIS_LOCAL/idris2-0.8.0${PACK_PKG_PATH:+:$PACK_PKG_PATH}"

check() { # <file>  -> echoes idris2 output, returns idris2 exit code
	local f="$1"
	( cd "$(dirname "$f")" && \
		IDRIS2_PACKAGE_PATH="$PKG_PATH" \
			"$IDRIS2" --check "$(basename "$f")" -p idris-ml -p idris-random -p linear -p contrib 2>&1 )
}

# ---- NEG: must fail, naming both mismatched dims ----
NEG_OUT="$(check "$NEG_FILE")"
NEG_RC=$?

if [ "$NEG_RC" -eq 0 ]; then
	echo "FAIL: negative test compiled cleanly — the Seq shape gate is" >&2
	echo "      broken. A mis-sized chain (256 -> 128) type-checked." >&2
	echo "" >&2
	echo "--- idris2 output (neg) ---" >&2
	echo "$NEG_OUT" >&2
	exit 1
fi

# Strip echoed source-context lines (` NN | ...`) before grepping — the
# fixture's comments quote the expected error text, so grepping the raw
# output would let a wrong-reason failure pass the gate.
NEG_ERRS="$(echo "$NEG_OUT" | grep -Ev '^[[:space:]]*[0-9]+ \|')"
if ! echo "$NEG_ERRS" | grep -q "ChainFits 256 128"; then
	echo "FAIL: negative test errored, but not with the dim-naming" >&2
	echo "      ChainFits error — the error-quality guarantee regressed" >&2
	echo "      (likely back to the opaque 'Module ?l' failure)." >&2
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
	echo "      trusted to fail for the right reason (the Seq/ChainFits" >&2
	echo "      surface itself may be broken)." >&2
	echo "" >&2
	echo "--- idris2 output (pos) ---" >&2
	echo "$POS_OUT" >&2
	exit 1
fi

echo "PASS: Seq shape gate — mis-sized chain fails naming both dims (ChainFits 256 128), well-sized chain compiles"
exit 0

#!/usr/bin/env bash
# Verifies that F1 (#412) `LosslessTo` REFUSES to resolve when an
# integer source's range overflows the target float's exact-integer
# range. Inverts the idris2 exit code (success = compile failed) and
# asserts the error mentions the impossible LTE proof.
#
# Direction tested: IntN 64 → Float 32. Max IntN 64 value 2^63 is
# far beyond F32's exact-integer ceiling of 2^24. The required
# `LTE 64 25` proof has no inhabitant.
#
# If this gate ever passes (i.e., the neg file compiles cleanly),
# the int → float lossless gate has regressed and silent mantissa-
# overflowing casts would be permitted — exactly the kind of
# precision loss idris-ml's type system is set up to refuse.
#
# Run from the repo root, after `make install`.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/src/Test/neg/IntOverflowToFloatRejected.idr"
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

# Success path: idris2 should have errored with the unsolvable LTE
# proof. Match on the literal "LTE 64 25" so unrelated regressions
# (e.g. a parse error) don't accidentally pass the gate.
if echo "$OUTPUT" | grep -q "IntOverflowToFloatRejected:" \
	 && echo "$OUTPUT" | grep -q "LTE 64 25"; then
	echo "PASS: int-overflow lossless-cast gate refuses I64 → F32"
	exit 0
fi

# Failure paths.
if echo "$OUTPUT" | grep -q "IntOverflowToFloatRejected:"; then
	echo "FAIL: negative test errored, but the error doesn't mention" >&2
	echo "      'LTE 64 25' — gate may have regressed to a different" >&2
	echo "      failure mode." >&2
else
	echo "FAIL: negative test compiled cleanly — the int-overflow" >&2
	echo "      lossless-cast gate is broken. LosslessTo accepts a" >&2
	echo "      mantissa-overflowing direction; silent I64 → F32 casts" >&2
	echo "      could slip past the type checker." >&2
fi

echo "" >&2
echo "--- idris2 output ---" >&2
echo "$OUTPUT" >&2
exit 1

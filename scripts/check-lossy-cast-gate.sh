#!/usr/bin/env bash
# Verifies that `LosslessTo` (in `DType.Core`) REFUSES to resolve for a
# known-lossy float-cast direction. Inverts the idris2 exit code
# (success = compile failed) and asserts the error message mentions
# the impossible LTE proof so we don't accept failures that come from
# unrelated regressions.
#
# The lossy direction tested is Float 32 → BFloat 16, which shrinks
# mantissa from 23 to 7. The required `LTE 23 7` proof is unsolvable.
#
# If this gate ever passes (i.e., the neg file compiles cleanly), the
# cross-family lossless-cast typeclass has regressed and silent mid-
# graph F32 → BF16 casts would be permitted — exactly the kind of
# precision loss idris-ml's type system is set up to refuse.
#
# Run from the repo root, after `make install`.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/src/Test/neg/LossyDirectionRejected.idr"
# IDRIS2_LOCAL is provided by the Makefile invoking this script. The
# multi-build-key refactor moved the installed-package prefix from
# `$REPO_ROOT/.idris2` to `$REPO_ROOT/build/<BUILD_KEY>/idris2-prefix`,
# so scripts can no longer hard-code the location.
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
					"$IDRIS2" --check "$(basename "$NEG_FILE")" -p idris-ml 2>&1 || true)"

# Success path: idris2 should have errored with the unsolvable LTE
# proof. Match on the literal "LTE 23 7" so unrelated regressions
# (e.g. a parse error) don't accidentally pass the gate.
if echo "$OUTPUT" | grep -q "LossyDirectionRejected:" \
	 && echo "$OUTPUT" | grep -q "LTE 23 7"; then
	echo "PASS: cross-family lossless-cast gate refuses F32 → BF16"
	exit 0
fi

# Failure paths.
if echo "$OUTPUT" | grep -q "LossyDirectionRejected:"; then
	echo "FAIL: negative test errored, but the error doesn't mention" >&2
	echo "      'LTE 23 7' — gate may have regressed to a different" >&2
	echo "      failure mode." >&2
else
	echo "FAIL: negative test compiled cleanly — the cross-family" >&2
	echo "      lossless-cast gate is broken. LosslessTo accepts a" >&2
	echo "      mantissa-shrinking direction; silent F32 → BF16 mid-" >&2
	echo "      graph casts could slip past the type checker." >&2
fi

echo "" >&2
echo "--- idris2 output ---" >&2
echo "$OUTPUT" >&2
exit 1

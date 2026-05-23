#!/usr/bin/env bash
# Verifies the linear-types discipline on `freezeNetwork` rejects reuse
# of the pre-freeze Network reference. Inverts the idris2 exit code and
# matches on "linear" or "uses" so we don't accept failures from
# unrelated regressions.
#
# Run from the repo root, after `make install`.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/test/neg/AliasAfterFreeze.idr"
# IDRIS2_LOCAL is provided by the Makefile invoking this script. The
# multi-build-key refactor moved the installed-package prefix from
# `$REPO_ROOT/.idris2` to `$REPO_ROOT/build/<BUILD_KEY>/idris2-prefix`,
# so scripts can no longer hard-code the location.
IDRIS_LOCAL="${IDRIS2_LOCAL:-$REPO_ROOT/.idris2}"

if [ ! -f "$NEG_FILE" ]; then
  echo "FAIL: negative test file missing at $NEG_FILE" >&2
  exit 1
fi

if [ ! -d "$IDRIS_LOCAL" ]; then
  echo "FAIL: $IDRIS_LOCAL not found — run 'make install' first (sets IDRIS2_LOCAL)" >&2
  exit 1
fi

cd "$(dirname "$NEG_FILE")"
OUTPUT="$(IDRIS2_PACKAGE_PATH="$IDRIS_LOCAL/idris2-0.8.0" \
          idris2 --check "$(basename "$NEG_FILE")" -p idris-ml 2>&1 || true)"

# Success path: idris2 should have produced a linearity error.
if echo "$OUTPUT" | grep -q "AliasAfterFreeze:" \
   && echo "$OUTPUT" | grep -qiE "linear|uses of"; then
  echo "PASS: aliasing rejected — freezeNetwork's linear discipline holds"
  exit 0
fi

# Failure paths.
if echo "$OUTPUT" | grep -q "AliasAfterFreeze:"; then
  echo "FAIL: negative test errored, but the error doesn't mention" >&2
  echo "      linearity / uses — discipline may have regressed to a" >&2
  echo "      different failure mode." >&2
else
  echo "FAIL: negative test compiled cleanly — freezeNetwork's linear" >&2
  echo "      consumption is broken. A user can freeze a network and" >&2
  echo "      then accidentally train via the original reference." >&2
fi

echo "" >&2
echo "--- idris2 output ---" >&2
echo "$OUTPUT" >&2
exit 1

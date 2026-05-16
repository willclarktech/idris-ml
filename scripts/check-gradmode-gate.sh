#!/usr/bin/env bash
# Verifies the GradMode gate rejects a NoGrad loss being passed to
# nativeTrainStep. Inverts the idris2 exit code (success = compile failed)
# and asserts the error message mentions both `WithGrad` and `NoGrad` so we
# don't accept failures that come from unrelated regressions.
#
# Run from the repo root, after `make install` (so idris-ml is in
# IDRIS2_PACKAGE_PATH at .idris2/).

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEG_FILE="$REPO_ROOT/packages/idris-ml/test/neg/GateRejectsNoGrad.idr"
IDRIS_LOCAL="$REPO_ROOT/.idris2"

if [ ! -f "$NEG_FILE" ]; then
  echo "FAIL: negative test file missing at $NEG_FILE" >&2
  exit 1
fi

if [ ! -d "$IDRIS_LOCAL" ]; then
  echo "FAIL: $IDRIS_LOCAL not found — run 'make install' first" >&2
  exit 1
fi

cd "$(dirname "$NEG_FILE")"
OUTPUT="$(IDRIS2_PACKAGE_PATH="$IDRIS_LOCAL/idris2-0.8.0" \
          idris2 --check "$(basename "$NEG_FILE")" -p idris-ml 2>&1 || true)"

# Success path: idris2 should have produced an error mentioning both
# GradMode constructors. If it compiled cleanly, the gate is broken.
if echo "$OUTPUT" | grep -q "GateRejectsNoGrad:" \
   && echo "$OUTPUT" | grep -q "WithGrad" \
   && echo "$OUTPUT" | grep -q "NoGrad"; then
  echo "PASS: gate rejects NoGrad loss with the expected type error"
  exit 0
fi

# Failure paths.
if echo "$OUTPUT" | grep -q "GateRejectsNoGrad:"; then
  echo "FAIL: negative test errored, but the error doesn't mention" >&2
  echo "      WithGrad / NoGrad — gate may have regressed to a" >&2
  echo "      different failure mode." >&2
else
  echo "FAIL: negative test compiled cleanly — the GradMode gate is" >&2
  echo "      broken. nativeTrainStep accepts a NoGrad loss; the" >&2
  echo "      'inference loss silently no-ops training' bug class" >&2
  echo "      is no longer caught." >&2
fi

echo "" >&2
echo "--- idris2 output ---" >&2
echo "$OUTPUT" >&2
exit 1

#!/usr/bin/env bash
# Gate: the fit epilogue (PERF_MS_PER_EP + the C profile report) is
# INFO-level output per Util/Log.idr's level scheme, so
# IDRISML_LOG_LEVEL=warn must silence it. PERF_MS_PER_EP goes through
# logInfo (C-side gated); the profile report needs the Idris-side
# `getLogLevel >= levelInfo` guard in Fit.idr — this script gates that
# guard. notebooks-refresh (mk/jupyter.mk) depends on the suppression
# for deterministic committed notebook outputs.
#
# Vehicle: the supervised example (tape backend, ~1 s).
# Usage: scripts/test-log-level-profile-gate.sh [supervised-binary]
set -euo pipefail

BIN="${1:-./build/exec/supervised}"

echo "=== Step 1: default level (info) — epilogue present ==="
OUT_INFO="$("$BIN" --epochs 3 --seed 7 2>&1)"
echo "$OUT_INFO" | grep -q "=== Profile Report ===" \
	|| { echo "FAIL: default-level run did not print the profile report"; exit 1; }
echo "$OUT_INFO" | grep -q "PERF_MS_PER_EP=" \
	|| { echo "FAIL: default-level run did not print PERF_MS_PER_EP"; exit 1; }

echo "=== Step 2: IDRISML_LOG_LEVEL=warn — epilogue suppressed ==="
OUT_WARN="$(IDRISML_LOG_LEVEL=warn "$BIN" --epochs 3 --seed 7 2>&1)"
if echo "$OUT_WARN" | grep -q "=== Profile Report ==="; then
	echo "FAIL: warn-level run printed the profile report"; exit 1
fi
if echo "$OUT_WARN" | grep -q "PERF_MS_PER_EP="; then
	echo "FAIL: warn-level run printed PERF_MS_PER_EP"; exit 1
fi

echo "PASS: fit profile epilogue respects the log level"

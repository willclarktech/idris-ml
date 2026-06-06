#!/usr/bin/env bash
# Smoke test for training-loop checkpoint/resume (Train.idr + Checkpoint.idr).
#
# Trains the gpt example (embedded corpus, fast) for 10 epochs into a
# checkpoint dir, then resumes to 20 and asserts that:
#   1. the trainer_state.json sidecar records epoch 10,
#   2. the resumed run logs "Resuming from epoch 10" (loaded the sidecar),
#   3. the resumed run completes at epochs=20 (continued, not restarted).
#
# Deterministic on the tape backend; gates against regressions in the
# save/resume path. Usage: scripts/test-checkpoint-resume.sh [gpt-binary]
set -euo pipefail

BIN="${1:-./build/exec/gpt}"
SEED=7
DIR="$(mktemp -d "${TMPDIR:-/tmp}/ckpt-resume.XXXXXX")"
trap 'rm -rf "$DIR"' EXIT

echo "=== Step 1: train 10 epochs (checkpoint every 5) → $DIR ==="
"$BIN" --epochs 10 --seed "$SEED" --checkpoint-dir "$DIR" --checkpoint-every 5

for f in last.model.safetensors last.opt.safetensors last.trainer_state.json \
				 best.model.safetensors best.opt.safetensors; do
	[ -f "$DIR/$f" ] || { echo "FAIL: missing checkpoint artifact $f"; exit 1; }
done

EP="$(grep -o '"epoch":[[:space:]]*[0-9]*' "$DIR/last.trainer_state.json" | grep -o '[0-9]*')"
echo "sidecar epoch=$EP"
[ "$EP" = "10" ] || { echo "FAIL: expected sidecar epoch 10, got '$EP'"; exit 1; }

echo "=== Step 2: resume to 20 epochs ==="
OUT="$("$BIN" --epochs 20 --seed "$SEED" --resume "$DIR" 2>&1)"
echo "$OUT"

echo "$OUT" | grep -q "Resuming from epoch 10" \
	|| { echo "FAIL: resumed run did not load the epoch-10 checkpoint"; exit 1; }
echo "$OUT" | grep -q "epochs=20" \
	|| { echo "FAIL: resumed run did not complete at 20 epochs"; exit 1; }

echo "PASS: checkpoint resume smoke test (epoch 10 → 20)"

#!/usr/bin/env bash
# tokenize-text-corpus.sh — tokenize a plain-text file via a HF
# AutoTokenizer and emit a flat comma-separated list of integer
# token IDs at the destination path.
#
# Usage:
#   bash scripts/tokenize-text-corpus.sh <text-file> <tokenizer> <out-path>
#
# Example:
#   bash scripts/tokenize-text-corpus.sh data/tinyshakespeare/input.txt \
#        distilgpt2 data/tinyshakespeare/input.distilgpt2.tokens
#
# Output format: a single line of comma-separated token IDs (no labels,
# no per-example structure — this is a corpus stream for LM tasks). The
# Idris-side `loadGpt2Tokens` reads it via readFile + split-on-comma +
# parseInteger.
#
# Dependencies: `transformers` (already in the pytorch uv venv).
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <text-file> <tokenizer> <out-path>" >&2
  exit 1
fi

TEXT_FILE=$(realpath "$1")
TOKENIZER=$2
OUT_PATH=$(realpath -m "$3")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd)

if [[ -s "$OUT_PATH" && "${HF_FORCE_REDOWNLOAD:-0}" != "1" ]]; then
  echo "tokenize-text-corpus: $OUT_PATH already present (set HF_FORCE_REDOWNLOAD=1 to refresh)"
  exit 0
fi

mkdir -p "$(dirname "$OUT_PATH")"

cd "$REPO_ROOT/packages/pytorch"
uv run python - "$TEXT_FILE" "$TOKENIZER" "$OUT_PATH" <<'PYEOF'
import sys
from transformers import AutoTokenizer

text_path, tokenizer_id, out_path = sys.argv[1:4]
print(f"tokenize-text-corpus: {tokenizer_id!r} -> {out_path!r}")

with open(text_path) as f:
    text = f.read()

tok = AutoTokenizer.from_pretrained(tokenizer_id)
# add_special_tokens=False — for an LM corpus we want a continuous token
# stream, not per-sentence [CLS]/[SEP] insertion. The model learns from
# the full sequence as a single document.
ids = tok.encode(text, add_special_tokens=False)
print(f"  source: {len(text)} chars  -> {len(ids)} tokens")

with open(out_path, "w") as f:
    f.write(",".join(str(i) for i in ids))
    f.write("\n")
print(f"  -> {out_path}")
PYEOF

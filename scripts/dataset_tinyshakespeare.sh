#!/bin/bash
# Download tinyshakespeare corpus (1.1 MB, 65-char vocab) to data/tinyshakespeare/.
# Source: karpathy/char-rnn — the canonical char-level LM benchmark file.
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)/data/tinyshakespeare"
mkdir -p "$DIR"

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT="$DIR/input.txt"

if [ -f "$OUT" ]; then
  echo "Already exists: $OUT"
  exit 0
fi

echo "Downloading tinyshakespeare ..."
curl -sL "$URL" -o "$OUT"
echo "  -> $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"

echo "tinyshakespeare ready in $DIR"

#!/usr/bin/env bash
# Download tinyshakespeare corpus (1.1 MB, 65-char vocab) to data/tinyshakespeare/.
# Source: karpathy/char-rnn — the canonical char-level LM benchmark file.
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)/data/tinyshakespeare"
mkdir -p "$DIR"

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT="$DIR/input.txt"

# `-s` (non-empty) not `-f` (exists) — catches half-downloaded files
# from interrupted curl runs that would otherwise look cached.
if [ -s "$OUT" ]; then
  echo "Already exists: $OUT"
  exit 0
fi

echo "Downloading tinyshakespeare ..."
curl -sL "$URL" -o "$OUT"
echo "  -> $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"

echo "tinyshakespeare ready in $DIR"

#!/usr/bin/env bash
# Download MNIST dataset (4 .idx files) to data/mnist/
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)/data/mnist"
mkdir -p "$DIR"

BASE="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
	"train-images-idx3-ubyte.gz"
	"train-labels-idx1-ubyte.gz"
	"t10k-images-idx3-ubyte.gz"
	"t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
	out="$DIR/${f%.gz}"
	# `-s` (non-empty) not `-f` (exists) — catches half-downloaded
	# files from interrupted curl runs that would otherwise look cached.
	if [ -s "$out" ]; then
		echo "Already exists: $out"
		continue
	fi
	echo "Downloading $f ..."
	curl -sL "$BASE/$f" | gunzip > "$out"
	echo "  -> $out ($(wc -c < "$out" | tr -d ' ') bytes)"
done

echo "MNIST data ready in $DIR"

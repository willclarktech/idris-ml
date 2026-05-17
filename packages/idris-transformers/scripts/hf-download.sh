#!/usr/bin/env bash
# hf-download.sh — fetch a HuggingFace safetensors checkpoint.
#
# Usage:
#   bash scripts/hf-download.sh <repo> [filename]
#
#   <repo>      HF Hub repo, e.g. `prajjwal1/bert-tiny` or `gpt2`.
#   [filename]  File to fetch. Default: `model.safetensors`.
#               If filename ends in `.index.json`, the script follows
#               the manifest's `weight_map` and downloads every unique
#               shard listed there.
#
# Examples:
#   bash scripts/hf-download.sh prajjwal1/bert-tiny
#   bash scripts/hf-download.sh sshleifer/tiny-gpt2 model.safetensors
#   bash scripts/hf-download.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#        model.safetensors.index.json
#
# Output:
#   Files land in packages/idris-transformers/models/<repo>/ relative
#   to the repo root. Directory is created if missing. Everything under
#   `models/` is gitignored — it's the local cache, not source.
#
# Env:
#   HF_TOKEN  Optional bearer token for private/gated models. If set,
#             `curl` includes `Authorization: Bearer $HF_TOKEN`.
#
# Dependencies:
#   - curl (with --fail and -L support)
#   - python3 (for JSON parsing when filename is .index.json)
#
# Exit codes:
#   0  success
#   1  bad arguments / missing dependency
#   2  HTTP error from Hub (curl --fail bubbled up)
set -euo pipefail

usage() {
  sed -n '1,/^set -euo/p' "$0" | sed -n '1,/^# Exit codes:/p'
  exit "${1:-1}"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "error: expected 1 or 2 args, got $#" >&2
  usage 1
fi

REPO=$1
FILENAME=${2:-model.safetensors}

# Resolve the package models cache directory relative to this script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PKG_DIR=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
DEST_DIR="$PKG_DIR/models/$REPO"
mkdir -p "$DEST_DIR"

# Build the curl invocation. Bash array so we can safely append the
# auth header when HF_TOKEN is set.
CURL=(curl -L --fail --silent --show-error)
if [[ -n "${HF_TOKEN:-}" ]]; then
  CURL+=(-H "Authorization: Bearer $HF_TOKEN")
fi

download_one() {
  local fname=$1
  local url="https://huggingface.co/$REPO/resolve/main/$fname"
  local dest="$DEST_DIR/$fname"
  mkdir -p "$(dirname "$dest")"
  echo "  fetching $url"
  "${CURL[@]}" -o "$dest" "$url"
}

case "$FILENAME" in
  *.index.json)
    # Sharded model: fetch the index, parse its weight_map, fetch each
    # unique shard filename listed there.
    if ! command -v python3 >/dev/null 2>&1; then
      echo "error: python3 not found; required to parse $FILENAME" >&2
      exit 1
    fi
    download_one "$FILENAME"
    INDEX_PATH="$DEST_DIR/$FILENAME"
    SHARDS=$(python3 -c "
import json, sys
with open('$INDEX_PATH') as f:
    idx = json.load(f)
shards = sorted(set(idx.get('weight_map', {}).values()))
print('\n'.join(shards))
")
    if [[ -z "$SHARDS" ]]; then
      echo "warning: weight_map in $FILENAME was empty" >&2
    fi
    while IFS= read -r shard; do
      [[ -z "$shard" ]] && continue
      download_one "$shard"
    done <<< "$SHARDS"
    ;;
  *)
    download_one "$FILENAME"
    ;;
esac

echo "done. files in: $DEST_DIR"

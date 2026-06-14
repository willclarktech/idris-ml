#!/usr/bin/env bash
# hf-download.sh — fetch a HuggingFace repo's safetensors weights + tokenizer
# + config files into <repo-root>/models/<repo>/.
#
# Usage:
#   bash scripts/hf-download.sh <repo>
#
#   <repo>  HF Hub repo, e.g. `distilgpt2` or `meta-llama/Llama-3.2-1B`.
#
# Output:
#   Files land at <repo-root>/models/<repo>/ in flat layout matching
#   HF on-disk (model.safetensors, config.json, tokenizer.json, ...).
#   Both Idris's loadModel and Python's AutoModel.from_pretrained read
#   from the same directory — no separate `~/.cache/huggingface/` copy.
#
# Env:
#   HF_TOKEN              Optional bearer token for private/gated repos.
#   HF_FORCE_REDOWNLOAD   If set to "1", force re-download even when
#                         huggingface_hub thinks the local copy is current.
#                         By default snapshot_download checks ETags and
#                         skips unchanged files (its own etag-based cache).
#
# Dependencies:
#   - Python `huggingface_hub` (provided by the pytorch package's uv venv).
#
# Why snapshot_download (not curl):
#   - One call grabs all needed files (config.json + model.safetensors +
#     tokenizer files) so Python's AutoModel.from_pretrained(local_dir)
#     can load directly without a separate ~/.cache copy.
#   - Built-in etag-based caching — re-running the script when files
#     are current is a no-op (no re-download, no re-write of files).
#   - Handles sharded models automatically (model.safetensors.index.json
#     + shards) without bespoke .index.json parsing.
#   - `ignore_patterns` skips the `original/` PyTorch-pickled mirrors
#     (Llama 3.2 1B's `original/consolidated.00.pth` is 2.4 GB — same
#     weights as the safetensors we DO download). Also skips .bin / .h5
#     / .msgpack mirrors. The package's stance is safetensors-only, so
#     these are pure waste.
set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "usage: $0 <repo>" >&2
	exit 1
fi

REPO=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd)
DEST_DIR="$REPO_ROOT/models/$REPO"

mkdir -p "$DEST_DIR"

FORCE_FLAG=""
if [[ "${HF_FORCE_REDOWNLOAD:-0}" == "1" ]]; then
	FORCE_FLAG="force_download=True,"
fi

# Run via the pytorch package's uv venv (huggingface_hub lives there
# alongside transformers). `cd`-ing in is the established pattern (see
# `save_oracle.py` invocation).
cd "$REPO_ROOT/packages/pytorch"
uv run python - "$REPO" "$DEST_DIR" <<PYEOF
import os, sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
print(f"hf-download: snapshot_download({repo!r}, local_dir={dest!r})")
path = snapshot_download(
		repo,
		local_dir=dest,
		# `or None`: an unset secret in CI maps to "" (not absent), and
		# snapshot_download("") emits an empty `Authorization: Bearer `
		# header → httpx LocalProtocolError. Empty ⇒ anonymous download
		# (correct for public/ungated repos like microsoft/bitnet-*).
		token=os.environ.get("HF_TOKEN") or None,
		${FORCE_FLAG}
		# Allow-list (not deny-list): explicitly grab only what
		# AutoModel.from_pretrained + AutoTokenizer.from_pretrained need.
		# Anything else (PT pickled mirrors, ONNX, CoreML, TFLite, Rust
		# tract .ot files, original/* PyTorch bundles, README, license)
		# stays in the cloud. Adding ".bin" / ".h5" / etc. to a deny-list
		# would still miss future formats; allow-list is forward-safe.
		allow_patterns=[
				# Weights — sharded or single-file.
				"*.safetensors",
				"*.safetensors.index.json",
				# Model configs.
				"config.json",
				"generation_config.json",
				# Tokenizer files (BERT WordPiece uses vocab.txt; GPT-2 BPE
				# uses vocab.json + merges.txt; modern HF uses tokenizer.json).
				"tokenizer.json",
				"tokenizer_config.json",
				"vocab.json",
				"vocab.txt",
				"merges.txt",
				"special_tokens_map.json",
		],
)
print(f"  -> {path}")
PYEOF

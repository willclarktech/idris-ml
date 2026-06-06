#!/usr/bin/env bash
# hf-download-dataset.sh — fetch a HuggingFace text-classification dataset,
# tokenize via a HF AutoTokenizer, and write a simple TSV at
# <repo-root>/data/hf-datasets/<dataset>/<split>.tsv.
#
# Usage:
#   bash scripts/hf-download-dataset.sh <dataset> <split> [<config>] [<tokenizer>]
#
#   <dataset>    HF Hub dataset id (e.g. `glue`, `imdb`).
#   <split>      e.g. `train`, `validation`, `test`.
#   <config>     Optional dataset subset / config (e.g. `sst2` for `glue`).
#                Omit (or pass "") for datasets without subsets.
#   <tokenizer>  Optional HF tokenizer id (default:
#                `google/bert_uncased_L-2_H-128_A-2` — matches FT3's example
#                backbone).
#
# Output format (per line, TAB-separated, two columns):
#
#   <label>\t<comma-separated token ids>
#
# Example line: `1\t101,7592,2003,1037,3835,3185,102`
#
# Rationale for the TSV (not JSONL or arrow): the Idris side reads it via
# `readFile` + `Data.String.split` — no JSON parser dependency. Token IDs
# fit comfortably under Idris-2's Integer cast at this corpus size.
#
# Env:
#   HF_TOKEN              Optional bearer token for private/gated datasets.
#   HF_FORCE_REDOWNLOAD   If set to "1", overwrite existing TSV.
#
# Dependencies:
#   - Python `datasets`, `transformers` (provided by the pytorch package's
#     uv venv — the same env that backs `hf-download.sh`).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <dataset> <split> [<config>] [<tokenizer>]" >&2
  exit 1
fi

DATASET=$1
SPLIT=$2
CONFIG=${3:-}
TOKENIZER=${4:-google/bert_uncased_L-2_H-128_A-2}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd)

# Bucket per dataset+config so glue/sst2 doesn't collide with glue/mrpc etc.
BUCKET="$DATASET"
[[ -n "$CONFIG" ]] && BUCKET="$DATASET-$CONFIG"
DEST_DIR="$REPO_ROOT/data/hf-datasets/$BUCKET"
mkdir -p "$DEST_DIR"
OUT_PATH="$DEST_DIR/$SPLIT.tsv"

if [[ -s "$OUT_PATH" && "${HF_FORCE_REDOWNLOAD:-0}" != "1" ]]; then
  echo "hf-download-dataset: $OUT_PATH already present (set HF_FORCE_REDOWNLOAD=1 to refresh)"
  exit 0
fi

# Run inside the pytorch package's uv venv (same as `hf-download.sh`).
cd "$REPO_ROOT/packages/pytorch"
uv run python - "$DATASET" "$SPLIT" "$CONFIG" "$TOKENIZER" "$OUT_PATH" <<'PYEOF'
import os, sys
from datasets import load_dataset
from transformers import AutoTokenizer

dataset, split, config, tokenizer_id, out_path = sys.argv[1:6]
print(f"hf-download-dataset: load_dataset({dataset!r}, {config or '<no-config>'}, split={split!r})")

ds_args = {"split": split}
if config:
    ds_args["name"] = config
ds = load_dataset(dataset, token=os.environ.get("HF_TOKEN"), **ds_args)

# Identify the text + label columns. SST-2 uses `sentence` + `label`; IMDb
# uses `text` + `label`. Fall back to the first string-ish column for text.
text_col = None
for cand in ("sentence", "text", "premise", "sentence1"):
    if cand in ds.column_names:
        text_col = cand
        break
if text_col is None:
    raise SystemExit(f"hf-download-dataset: no recognised text column in "
                     f"{ds.column_names}")
if "label" not in ds.column_names:
    raise SystemExit(f"hf-download-dataset: no 'label' column in "
                     f"{ds.column_names}")

print(f"  text column: {text_col!r}, label column: 'label', N = {len(ds)}")
print(f"  tokenizing via {tokenizer_id!r}")
tok = AutoTokenizer.from_pretrained(tokenizer_id)

with open(out_path, "w") as out:
    for ex in ds:
        text  = ex[text_col]
        label = int(ex["label"])
        # No truncation here — the Idris-side `padToSeqLen` decides the
        # working seqLen and trims/pads per batch. Adds [CLS] / [SEP] via
        # the tokenizer's default `add_special_tokens=True`.
        ids = tok.encode(text)
        out.write(f"{label}\t{','.join(str(i) for i in ids)}\n")
print(f"  -> {out_path}")
PYEOF

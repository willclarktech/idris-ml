"""Produce the integration-test oracle for Example/HfBertInference.

Loads `google/bert_uncased_L-2_H-128_A-2` via HuggingFace transformers,
runs forward on a fixed input, and writes the pooled `[CLS]` output to
`models/bert-tiny-oracle.safetensors`. The Idris side reads the same
file and asserts max-abs-difference against its own forward output
within F32 tolerance.

The oracle lives alongside the downloaded checkpoints under `models/`
because both serve the same workflow — `models/` is the gitignored
local cache for everything we pull from / generate against the Hub.

This script is the canonical source of truth for the oracle. When
HfBert.idr's forward semantics change, regenerate the fixture and the
hash drift is exactly the divergence to investigate.

Usage:
    cd packages/pytorch && uv run python -m \\
        idris_transformers.scripts.save_oracle  # if module-style

    # or invoked directly via the Makefile target:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

# Resolve the models cache directory relative to this script regardless
# of where the user `cd`'d before invoking. The path is documented in
# Example/HfBertInference.idr so the Idris side reads from the same
# location.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent   # <repo-root>
MODELS_DIR = REPO_ROOT / "models"
ORACLE_PATH = MODELS_DIR / "bert-tiny-oracle.safetensors"

# The model we anchor the oracle against. Same hidden=128, layers=2,
# intermediate=512 family as prajjwal1/bert-tiny but ships safetensors.
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
# Local copy populated by `hf-download.sh` (which calls
# huggingface_hub.snapshot_download). Same files Idris's loadModel reads
# — single physical copy on disk, no separate `~/.cache/huggingface/`.
MODEL_LOCAL = MODELS_DIR / MODEL_ID

# Fixed input. WordPiece tokenization of "hello" under BERT's
# bert-base-uncased vocab yields [CLS]=101 hello=7592 [SEP]=102. The
# Idris side hardcodes the same IDs (no tokenizer in idris-transformers
# v1 — see Row 7's LLM-class example for the tokenizer integration).
FIXED_INPUT_IDS: list[int] = [101, 7592, 102]
HIDDEN: int = 128  # bert-tiny hidden size; matches the model config


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Determinism: BERT inference has no stochastic ops at eval time
    # (no dropout, no sampling), so a fixed seed isn't strictly
    # required — but setting it is cheap insurance against future
    # changes that might add nondeterminism.
    torch.manual_seed(42)

    print(f"loading {MODEL_ID} from {MODEL_LOCAL} ...")
    assert MODEL_LOCAL.is_dir(), (
        f"{MODEL_LOCAL} not found — run `bash packages/idris-transformers/"
        f"scripts/hf-download.sh {MODEL_ID}` first (or `make example-hf-bert-"
        f"inference` which depends on it via the Makefile pattern rule)."
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LOCAL))
    model = AutoModel.from_pretrained(str(MODEL_LOCAL))
    model.eval()

    # Sanity-check the tokenization matches our hardcoded IDs. If the
    # tokenizer changes upstream (vocab swap, casing fix), this
    # surfaces the divergence loudly.
    actual_ids = tokenizer.encode("hello", add_special_tokens=True)
    assert actual_ids == FIXED_INPUT_IDS, (
        f"Tokenizer drift: expected {FIXED_INPUT_IDS}, got {actual_ids}. "
        f"The Idris side hardcodes {FIXED_INPUT_IDS}; either regenerate "
        f"both sides or pin the tokenizer version."
    )

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long)  # [batch=1, seq=3]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # The pooler output is the [CLS] token after the final pooler dense
    # + tanh. Shape: [batch=1, hidden=128].
    pooled = outputs.pooler_output
    assert pooled.shape == (1, HIDDEN), f"pooler shape {pooled.shape} != [1, {HIDDEN}]"
    assert torch.isfinite(pooled).all(), "pooler output contains non-finite values"

    # Strip the batch dim so the Idris side compares a flat [128] vector.
    output_vec = pooled.squeeze(0).contiguous()
    assert output_vec.shape == (HIDDEN,), f"flatten broke: {output_vec.shape}"

    save_file({"output": output_vec}, str(ORACLE_PATH))
    print(f"wrote {ORACLE_PATH}")
    print(f"  shape: {list(output_vec.shape)}")
    print(f"  dtype: {output_vec.dtype}")
    print(f"  first 5 values: {output_vec[:5].tolist()}")


if __name__ == "__main__":
    main()

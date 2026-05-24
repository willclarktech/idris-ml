"""Produce the integration-test oracle for Example/HfLlamaInference.

Loads `meta-llama/Llama-3.2-1B` via HuggingFace transformers, runs
forward on a fixed single-token input, and writes the last-position
hidden state (post final RmsNorm) to
`models/llama-3.2-1b-oracle.safetensors`. The Idris side reads the same
file and asserts max-abs-difference against its own forward output.

Llama 3.2 1B: vocab=128256, hidden=2048, n_layer=16, n_head=32,
n_kv_heads=8 (GQA 4:1), head_dim=64, intermediate=8192,
rope_base=500000, rms_norm_eps=1e-5, tie_word_embeddings=true. On-disk
~2.5 GB BF16; we cast to F32 to match the Idris torch-mps / mlx-gpu
example lanes.

The Idris `--dump-final-hidden` mode runs a single forward on the
hardcoded token id 9906 (= BPE("Hello") in Llama 3 vocab) and prints
the [hidden=2048] last-position hidden state to stdout. This oracle
mirrors that exactly.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle_llama.py

    # or via the Makefile:
    make test-transformers-oracle-llama

Pre-requisite: the model must be downloaded under
`<repo>/models/meta-llama/Llama-3.2-1B/` (license-gated; fetch with
HF_TOKEN=hf_... bash packages/idris-transformers/scripts/hf-download.sh
meta-llama/Llama-3.2-1B).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
ORACLE_PATH = MODELS_DIR / "llama-3.2-1b-oracle.safetensors"
MODEL_LOCAL = MODELS_DIR / "meta-llama" / "Llama-3.2-1B"

MODEL_ID = "meta-llama/Llama-3.2-1B"

# "Hello" under Llama 3 BPE = [9906]. Single token to keep the compute
# cheap on this 1.24B-param model (a single forward at hidden=2048
# already lands in the tens of seconds on torch-mps F32). The tokenizer
# drift assertion below catches any upstream tokenizer change.
FIXED_INPUT_IDS: list[int] = [9906]
HIDDEN: int = 2048


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    print(f"loading {MODEL_ID} from {MODEL_LOCAL} ...")
    assert MODEL_LOCAL.is_dir(), (
        f"{MODEL_LOCAL} not found — run `bash packages/idris-transformers/"
        f"scripts/hf-download.sh {MODEL_ID}` first (requires HF_TOKEN + "
        f"Llama 3.2 license acceptance)."
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LOCAL))
    # F32 on CPU: matches what the Idris torch-cpu / torch-mps F32 lanes
    # see. The oracle wants deterministic numerics, not throughput, so
    # CPU is fine even though the example runs on MPS/GPU.
    model = AutoModel.from_pretrained(str(MODEL_LOCAL), dtype=torch.float32)
    model.train(False)

    cfg = model.config
    print(
        f"  config: vocab_size={cfg.vocab_size} hidden_size={cfg.hidden_size} "
        f"num_hidden_layers={cfg.num_hidden_layers} "
        f"num_attention_heads={cfg.num_attention_heads} "
        f"num_key_value_heads={cfg.num_key_value_heads} "
        f"head_dim={cfg.hidden_size // cfg.num_attention_heads}"
    )
    assert cfg.hidden_size == HIDDEN, (
        f"HIDDEN={HIDDEN} disagrees with model hidden_size={cfg.hidden_size}. "
        f"Update the oracle constant or check the checkpoint."
    )

    # Confirm tokenization matches the hardcoded ID the Idris side uses
    # in --dump-final-hidden mode. Llama 3's BPE encodes "Hello" as a
    # single token id 9906 with no BOS prepended in this single-token
    # form. If HF rebuilds the tokenizer this surfaces the drift loudly.
    actual_ids = tokenizer.encode("Hello", add_special_tokens=False)
    assert actual_ids == FIXED_INPUT_IDS, (
        f"Tokenizer drift: expected {FIXED_INPUT_IDS}, got {actual_ids}."
    )

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    last_hidden = outputs.last_hidden_state  # [1, 1, 2048]
    assert last_hidden.shape == (1, len(FIXED_INPUT_IDS), HIDDEN), (
        f"last_hidden_state shape {last_hidden.shape} != "
        f"[1, {len(FIXED_INPUT_IDS)}, {HIDDEN}]"
    )
    assert torch.isfinite(last_hidden).all()

    # Last position; with a single input token this IS the only
    # position, but the slice mirrors the GPT-2 / BitNet oracle shape
    # so the comparator script handles the same [hidden]-shaped output
    # uniformly.
    output_vec = last_hidden[0, -1, :].contiguous()
    assert output_vec.shape == (HIDDEN,)

    save_file({"output": output_vec}, str(ORACLE_PATH))
    print(f"wrote {ORACLE_PATH}")
    print(f"  shape: {list(output_vec.shape)}")
    print(f"  dtype: {output_vec.dtype}")
    print(f"  first 5 values: {output_vec[:5].tolist()}")


if __name__ == "__main__":
    main()

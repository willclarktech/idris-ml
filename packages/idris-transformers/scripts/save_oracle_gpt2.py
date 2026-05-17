"""Produce the integration-test oracle for Example/HfGpt2Inference.

Loads `sshleifer/tiny-gpt2` via HuggingFace transformers, runs forward
on a fixed input, and writes the final hidden state for the last
sequence position to `models/tiny-gpt2-oracle.safetensors`. The Idris
side reads the same file and asserts max-abs-difference against its
own forward output within F32 tolerance.

`sshleifer/tiny-gpt2` is the canonical CI fixture HF themselves use to
exercise GPT-2 code paths without paying for a full-size model. It
ships safetensors and exercises every architectural piece of the
GPT-2 family (fused QKV via `c_attn.weight`, Conv1D transpose storage,
learned positional embeddings, causal masking, tied LM head).

Unlike BERT (where the natural gate is the pooler output), GPT-2 has
no pooler — the equivalent gate is the last position's final hidden
state, which threads through all 2 decoder blocks + the final
LayerNorm. The LM head's tied-decoder math is exercised separately by
the example's generation demo; gating on logits would inflate the
oracle to 50257 floats with no extra signal beyond what
[`hidden_size=32`] already provides.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle_gpt2.py

    # or via the Makefile:
    make test-transformers-oracle-gpt2
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent  # packages/idris-transformers/
MODELS_DIR = PKG_DIR / "models"
ORACLE_PATH = MODELS_DIR / "tiny-gpt2-oracle.safetensors"

MODEL_ID = "sshleifer/tiny-gpt2"

# Fixed input. "Hello world" under GPT-2 BPE = [15496, 995]. Two tokens
# is the minimum that exercises both the learned positional embedding
# (positions 0 and 1) AND the causal mask (position 1 attends to 0 but
# position 0 cannot attend to 1). Single-token inputs would mask half
# the GPT-2 surface from the gate.
FIXED_INPUT_IDS: list[int] = [15496, 995]

# `sshleifer/tiny-gpt2` is HF's own CI fixture for GPT-2 — intentionally
# minuscule (hidden=2, n_head=2, n_layer=2, head_dim=1). This is
# degenerate by ML standards but exercises every architectural piece
# the GPT-2 module needs to get right: fused QKV via c_attn, Conv1D
# transpose storage, learned positional embedding, causal mask, tied
# LM head. The Idris side will assert these dims at construction.
HIDDEN: int = 2


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # No stochastic ops at eval-time; seed is cheap insurance.
    torch.manual_seed(42)

    print(f"loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    # Print the config dims so the Idris side has a single source of
    # truth. The Idris example pins these at the type level.
    cfg = model.config
    print(
        f"  config: vocab_size={cfg.vocab_size} n_embd={cfg.n_embd} "
        f"n_layer={cfg.n_layer} n_head={cfg.n_head} "
        f"n_positions={cfg.n_positions} "
        f"head_dim={cfg.n_embd // cfg.n_head}"
    )
    assert cfg.n_embd == HIDDEN, (
        f"HIDDEN={HIDDEN} disagrees with model n_embd={cfg.n_embd}. "
        f"Update the oracle constant or check the checkpoint."
    )

    # Sanity-check tokenization matches our hardcoded IDs. If the
    # tokenizer changes upstream (vocab rebuild, special-token fix),
    # this surfaces the divergence loudly.
    actual_ids = tokenizer.encode("Hello world", add_special_tokens=True)
    assert actual_ids == FIXED_INPUT_IDS, (
        f"Tokenizer drift: expected {FIXED_INPUT_IDS}, got {actual_ids}. "
        f"The Idris side hardcodes {FIXED_INPUT_IDS}; either regenerate "
        f"both sides or pin the tokenizer version."
    )

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long)  # [batch=1, seq=2]

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # last_hidden_state shape: [batch=1, seq=2, hidden=32].
    last_hidden = outputs.last_hidden_state
    assert last_hidden.shape == (1, len(FIXED_INPUT_IDS), HIDDEN), (
        f"last_hidden_state shape {last_hidden.shape} != "
        f"[1, {len(FIXED_INPUT_IDS)}, {HIDDEN}]"
    )
    assert torch.isfinite(last_hidden).all(), "last_hidden contains non-finite values"

    # The gate compares the LAST position only — it carries information
    # from all earlier positions through causal attention, so a per-layer
    # bug surfaces here. Earlier positions would mask later-block bugs
    # because their attention is restricted.
    output_vec = last_hidden[0, -1, :].contiguous()
    assert output_vec.shape == (HIDDEN,), f"flatten broke: {output_vec.shape}"

    save_file({"output": output_vec}, str(ORACLE_PATH))
    print(f"wrote {ORACLE_PATH}")
    print(f"  shape: {list(output_vec.shape)}")
    print(f"  dtype: {output_vec.dtype}")
    print(f"  first 5 values: {output_vec[:5].tolist()}")


if __name__ == "__main__":
    main()

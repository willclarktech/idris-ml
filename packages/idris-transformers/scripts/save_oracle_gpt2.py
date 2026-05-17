"""Produce the integration-test oracle for Example/HfGpt2Inference.

Loads `hf-internal-testing/tiny-random-gpt2` via HuggingFace
transformers, runs forward on a fixed input, and writes the final
hidden state for the last sequence position to
`models/tiny-gpt2-oracle.safetensors`. The Idris side reads the same
file and asserts max-abs-difference against its own forward output
within F32 tolerance.

`hf-internal-testing/tiny-random-gpt2` is HF's random-init test
fixture — vocab=1000, hidden=32, n_layer=5, n_head=4, head_dim=8,
intermediate=128 (= 4 * hidden, GPT-2 default; the `intermediate_size: 37` in config.json is a stray BERT field GPT-2 ignores when n_inner is null), max_pos=512. It ships safetensors on disk (unlike
`sshleifer/tiny-gpt2` which only has pytorch_model.bin), so the
existing `hf-download.sh` fetches it without runtime conversion.
The architecture is the same as a "real" GPT-2 (fused QKV via
`c_attn.weight`, Conv1D transpose storage, learned positional
embeddings, causal masking, tied LM head); the weights are random so
generated text is gibberish, but the gate is "Idris forward matches
Python forward bit-for-byte within F32 tolerance" — pretraining
doesn't matter.

Unlike BERT (where the natural gate is the pooler output), GPT-2 has
no pooler — the equivalent gate is the last position's final hidden
state, which threads through all 5 decoder blocks + the final
LayerNorm. The LM head's tied-decoder math is exercised separately
by the example's generation demo (Phase-2 follow-up); gating on
logits would inflate the oracle to 1000 floats with no extra signal
beyond what `hidden=32` already provides.

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
from transformers import AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent  # packages/idris-transformers/
MODELS_DIR = PKG_DIR / "models"
ORACLE_PATH = MODELS_DIR / "tiny-gpt2-oracle.safetensors"

# `hf-internal-testing/tiny-random-gpt2` is HF's own random-init GPT-2
# fixture — vocab=1000, hidden=32, n_layer=5, n_head=4, head_dim=8,
# intermediate=128 (= 4 * hidden, GPT-2 default; the `intermediate_size: 37` in config.json is a stray BERT field GPT-2 ignores when n_inner is null), max_pos=512. Random weights (not pretrained — the
# `tiny-random-` prefix is the giveaway), but architecturally complete
# and CRITICALLY ships safetensors on disk (unlike sshleifer/tiny-gpt2
# which only has pytorch_model.bin and would need a runtime conversion).
# Small enough to be a cheap CI fixture; large enough that all the
# architectural pieces (fused QKV split, Conv1D transpose, multi-head
# causal attention via axis=1 narrow, GELU MLP, tied LM head) actually
# do meaningful work.
MODEL_ID = "hf-internal-testing/tiny-random-gpt2"

# Fixed input within the 1000-token random vocab. Two tokens minimum
# to exercise both the learned positional embedding (positions 0 and 1)
# AND the causal mask (position 1 attends to 0 but position 0 cannot
# attend to 1). Single-token inputs would mask half the GPT-2 surface.
# Token IDs picked from the vocab range — exact strings don't matter
# for a random-init model.
FIXED_INPUT_IDS: list[int] = [42, 137]
HIDDEN: int = 32


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # No stochastic ops at eval-time; seed is cheap insurance.
    torch.manual_seed(42)

    print(f"loading {MODEL_ID} ...")
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

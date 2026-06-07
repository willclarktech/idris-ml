"""Produce the integration-test oracle for Example/HfGpt2Inference.

Loads `distilgpt2` via HuggingFace transformers, runs forward on a fixed
input, and writes the last-position hidden state (post `ln_f`) to
`models/distilgpt2-oracle.safetensors`. The Idris side reads the same
file and asserts max-abs-difference against its own forward output.

`distilgpt2` (~350 MB safetensors) is real GPT-2 — pretrained, 6 layers
(half of GPT-2 small's 12), hidden=768, n_head=12, head_dim=64,
intermediate=3072, n_positions=1024, vocab=50257. Same naming
conventions as gpt2/gpt2-medium/gpt2-large/gpt2-xl, so HfGpt2 covers
the whole family.

Switched from `hf-internal-testing/tiny-random-gpt2` (random-init,
hidden=32, no meaningful generation) on 2026-05-26: the user-facing
demo needs a pretrained model to produce real text. Element-wise
hidden-state diff over distilgpt2's 6 layers + 768 hidden runs at
~1e-4 in practice (F32 numerics; same as BERT was tracking at). The
gate keeps the same shape — only the dims + checkpoint change.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle_gpt2.py

    # or via the Makefile:
    make test-transformers-oracle-gpt2
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

# safetensors' stubs type save_file's `filename` as str | PathLike[Unknown]
# (unparameterized PathLike), so the symbol is partially unknown to pyright;
# calls with a plain str are fine at runtime.
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]
from transformers import AutoModel, AutoTokenizer, GPT2Model, PreTrainedTokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # <repo-root>
MODELS_DIR = REPO_ROOT / "models"
ORACLE_PATH = MODELS_DIR / "distilgpt2-oracle.safetensors"
# Local model dir populated by hf-download.sh (snapshot_download).
MODEL_LOCAL = MODELS_DIR / "distilgpt2"

MODEL_ID = "distilgpt2"

# "Hello world" under GPT-2 BPE = [15496, 995]. Two tokens minimum to
# exercise the learned positional embedding (positions 0 + 1) AND the
# causal mask (position 1 attends to 0 but position 0 cannot attend to
# 1). The tokenizer drift assertion below will catch any upstream
# tokenizer change.
FIXED_INPUT_IDS: list[int] = [15496, 995]
HIDDEN: int = 768


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # torch leaves manual_seed's `seed` parameter unannotated, so the
    # member is partially unknown to pyright; fine at runtime.
    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]

    print(f"loading {MODEL_ID} from {MODEL_LOCAL} ...")
    assert MODEL_LOCAL.is_dir(), (
        f"{MODEL_LOCAL} not found — run `bash packages/idris-transformers/"
        f"scripts/hf-download.sh {MODEL_ID}` first."
    )
    # Auto*.from_pretrained is untyped (Unknown / loose union) in
    # transformers 5.x's stubs; the checkpoint is GPT-2, so pin the
    # concrete classes via cast — no behavior change.
    tokenizer = cast(
        "PreTrainedTokenizerFast",
        AutoTokenizer.from_pretrained(str(MODEL_LOCAL)),  # pyright: ignore[reportUnknownMemberType]
    )
    model = cast(
        "GPT2Model",
        AutoModel.from_pretrained(str(MODEL_LOCAL)),  # pyright: ignore[reportUnknownMemberType]
    )
    model.eval()

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

    # Confirm tokenization matches the hardcoded IDs the Idris side
    # uses in --dump-final-hidden mode. If HF rebuilds the BPE vocab,
    # this surfaces the drift loudly.
    # encode's **kwargs are unannotated in transformers 5.x, so the
    # member is partially unknown to pyright; the list[int] return is
    # typed and the call is fine at runtime.
    actual_ids: list[int] = tokenizer.encode("Hello world", add_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
    assert actual_ids == FIXED_INPUT_IDS, (
        f"Tokenizer drift: expected {FIXED_INPUT_IDS}, got {actual_ids}."
    )

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    last_hidden = outputs.last_hidden_state  # [1, 2, 768]
    assert last_hidden.shape == (1, len(FIXED_INPUT_IDS), HIDDEN), (
        f"last_hidden_state shape {last_hidden.shape} != [1, {len(FIXED_INPUT_IDS)}, {HIDDEN}]"
    )
    assert torch.isfinite(last_hidden).all()

    # Last position carries info from all earlier positions through
    # causal attention. Earlier positions would mask later-block bugs.
    output_vec = last_hidden[0, -1, :].contiguous()
    assert output_vec.shape == (HIDDEN,)

    save_file({"output": output_vec}, str(ORACLE_PATH))
    print(f"wrote {ORACLE_PATH}")
    print(f"  shape: {list(output_vec.shape)}")
    print(f"  dtype: {output_vec.dtype}")
    print(f"  first 5 values: {output_vec[:5].tolist()}")


if __name__ == "__main__":
    main()

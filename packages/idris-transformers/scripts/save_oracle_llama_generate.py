"""Produce the integration-test oracle for the Example/HfLlamaInference
greedy-generation path.

Companion to `save_oracle_llama.py`. Where that oracle covers a *single*
forward pass on `[9906]` ("Hello") and saves the [hidden]-shape final
hidden state for a max-abs-diff gate, this one covers *multi-token
greedy generation* and saves the resulting token-ID sequence (int64)
for an exact-match gate.

Loads `unsloth/Llama-3.2-1B` via HuggingFace transformers, tokenizes
a fixed prompt, runs `model.generate(do_sample=False, use_cache=True)`
for a fixed budget, and writes the full output sequence (prompt +
generated tokens) to
`models/llama-3.2-1b-generate-oracle.safetensors` with key
`"token_ids"`. The Idris side's `--dump-tokens` mode reads the same
prompt + budget, dumps the resulting tokens one Nat per line, and the
comparator (`compare_inference.py --token-sequence`) asserts
element-wise equality. Discrete tokens, no tolerance.

This guards the entire forward stack against drift: BPE tokenization,
embedding lookup, all 16 decoder blocks (RMSNorm + GQA-SDPA + SwiGLU),
final RMSNorm, tied LM head, argmax selection. Any per-position
disagreement fires the gate immediately, and since `--use-cache=True`
on the HF side matches "Idris no-cache" and (post-Phase-C) "Idris with
cache" mathematically, the gate is invariant to the KV cache landing
or being absent.

Prompt / budget choice: "The capital of France is" + 4 new tokens.
Matches the user-facing default in `runGenerate` so the gate exercises
the same path users hit. Budget 4 keeps the CI wall-clock manageable
on the torch-cpu lane (~4 forwards on growing sequences) while still
exercising multi-step generation. Bump to 8+ once KV cache lands.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle_llama_generate.py

    # or via the Makefile:
    make test-transformers-oracle-llama-generate

Pre-requisite: the model must be downloaded under
`<repo>/models/unsloth/Llama-3.2-1B/` (public mirror of Meta's weights;
no HF_TOKEN required; fetch with `bash
packages/idris-transformers/scripts/hf-download.sh
unsloth/Llama-3.2-1B`).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
ORACLE_PATH = MODELS_DIR / "llama-3.2-1b-generate-oracle.safetensors"
MODEL_LOCAL = MODELS_DIR / "unsloth" / "Llama-3.2-1B"

MODEL_ID = "unsloth/Llama-3.2-1B"

# Matches `runGenerate` default in HfLlamaInference.idr — the prompt the
# user-facing demo uses. The gate exercises the exact path the user
# would hit when running `make example-hf-llama-inference` with no
# extra args.
PROMPT = "The capital of France is"

# Budget. Bumped 2026-06-04 from 4 to 8 after the KV cache landed —
# with cached decode each step is constant-cost in Q/K/V projections
# (vs the no-cache path's growing prefix), so 8 tokens is cheap.
NUM_NEW_TOKENS = 8


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
    # F32 on CPU: deterministic numerics. Matches what the Idris
    # torch-cpu CI lane sees.
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_LOCAL), dtype=torch.float32)
    model.train(False)

    cfg = model.config
    print(
        f"  config: vocab_size={cfg.vocab_size} hidden_size={cfg.hidden_size} "
        f"num_hidden_layers={cfg.num_hidden_layers}"
    )

    # Tokenize the prompt the same way the Idris-side Tokenizer
    # subprocess does. `hf_tokenize.py` always calls
    # `tokenizer.encode(text, add_special_tokens=True)`, which for
    # Llama-3 prepends BOS `<|begin_of_text|>` (id 128000). To stay
    # apples-to-apples with the Idris-side generation path (which
    # consumes the subprocess output), match it here.
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f"  prompt:       {PROMPT!r}")
    print(f"  prompt ids:   {prompt_ids} (len={len(prompt_ids)})")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        # Greedy: do_sample=False, no temperature mixing. use_cache=True
        # is HF's default and matches the post-Phase-C Idris path; it's
        # mathematically equivalent to use_cache=False (same argmax,
        # same forward math, just faster).
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=NUM_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            temperature=1.0,
            pad_token_id=cfg.eos_token_id if cfg.eos_token_id is not None else 0,
        )

    # `out` is [1, p + N]; squeeze the batch dim and assert the length
    # is exactly p + N (no early-stop on EOS for this short budget).
    assert out.shape[0] == 1, f"batch shape {out.shape} unexpected"
    tokens = out[0].to(torch.int64).contiguous()
    expected_len = len(prompt_ids) + NUM_NEW_TOKENS
    assert tokens.shape == (expected_len,), (
        f"output length {tokens.shape[0]} != expected {expected_len}; "
        f"early-stop or padding interference. tokens={tokens.tolist()}"
    )
    # Sanity: prompt prefix must round-trip.
    prefix = tokens[: len(prompt_ids)].tolist()
    assert prefix == prompt_ids, f"prompt prefix drift: input {prompt_ids} → output[:p] {prefix}"

    save_file({"token_ids": tokens}, str(ORACLE_PATH))
    print(f"wrote {ORACLE_PATH}")
    print(f"  shape:        {list(tokens.shape)}")
    print(f"  dtype:        {tokens.dtype}")
    print(f"  full ids:     {tokens.tolist()}")
    print(f"  decoded:      {tokenizer.decode(tokens)!r}")


if __name__ == "__main__":
    main()

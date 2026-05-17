"""One-off head-to-head timing of Llama-3.2-1B inference in PyTorch
against the same prompt + decode budget Idris's HfLlamaInference uses.

Usage (from repo root):

    cd packages/pytorch && uv run python ../idris-transformers/scripts/time_inference_llama.py

Stage timings mirror the [stage] prints in
`packages/idris-ml-examples/src/Example/HfLlamaInference.idr`:

  tokenizer probe       — load AutoTokenizer
  model construction    — empty AutoModelForCausalLM (no weights)
  loadModelAllowCast    — load_state_dict from local safetensors
  greedy decode         — 8 tokens at prompt "The capital of France is"

Runs two configurations so the comparison is honest:

  (a) default               — HF's `model.generate`, uses internal KV cache
                              (this is what a PyTorch user actually pays)
  (b) no-cache (apples)     — disable use_cache so each new token reruns the
                              full forward on the growing sequence; matches
                              Idris's HfLlamaInference which has no KV cache
                              (TODO row "HfLlama KV cache + Llama 3.2 1B
                              end-to-end gate").

Output is one stage per line, formatted as `[stage] [hh:mm:ss] <label>`
so it visually matches the Idris-side log.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
MODEL_DIR  = REPO_ROOT / "models" / "meta-llama" / "Llama-3.2-1B"

PROMPT     = "The capital of France is"
NUM_TOKENS = 8
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE      = torch.float32   # matches Idris's torch-mps F32 lane


def fmt_elapsed(t0: float, now: float) -> str:
    secs = int(now - t0)
    return f"[{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}]"


def stamp(label: str, t0: float) -> None:
    print(f"[stage] {fmt_elapsed(t0, time.perf_counter())} {label}", flush=True)


def run_one(use_cache: bool, label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    stamp("tokenizer probe ok", t0)

    # `from_pretrained` does construction + load in one step in transformers'
    # default path. To split the two like Idris does, we'd need `from_config`
    # + `load_state_dict`, but the transformers loader has hooks that conflate
    # them; we instead time `from_pretrained` as a single "construct + load"
    # entry that subsumes both stage 2 + stage 3 in the Idris breakdown, and
    # note this in the output.
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,  # match Idris which doesn't lazy-load
    ).to(DEVICE)
    model.eval()
    stamp("hfLlamaModel + loadModelAllowCast ok (combined)", t0)

    # Tokenize prompt + greedy decode.
    inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=NUM_TOKENS,
            do_sample=False,
            use_cache=use_cache,
        )
    stamp(f"runGenerate done ({NUM_TOKENS} tokens, use_cache={use_cache})", t0)

    text = tok.decode(out[0], skip_special_tokens=False)
    print(f"Output: {text}", flush=True)


def main() -> int:
    print(f"Device: {DEVICE}, dtype: {DTYPE}", flush=True)
    print(f"Model:  {MODEL_DIR}", flush=True)
    print(f"Prompt: {PROMPT!r}, tokens: {NUM_TOKENS}", flush=True)

    if not MODEL_DIR.is_dir():
        print(f"\nERR: {MODEL_DIR} not found — run the hf-download.sh script first.", file=sys.stderr)
        return 1

    run_one(use_cache=True,  label="PyTorch — default (KV cache on)")
    run_one(use_cache=False, label="PyTorch — no-KV-cache (apples-to-Idris)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

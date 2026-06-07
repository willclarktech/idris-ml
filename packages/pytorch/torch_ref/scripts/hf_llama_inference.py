"""HuggingFace Llama inference Axis D reference.

PyTorch counterpart to packages/idris-ml-examples/src/Example/HfLlamaInference.idr.
Greedy-decode 8 tokens from a fixed prompt through Llama-3.2-1B and
emit PERF_GENERATE_TOKENS / PERF_GENERATE_WALL_MS markers.

Loads in F32 (matches the Idris F32 path); F16/BF16 reference is a
separate row. The wall covers prompt tokenize + 8 greedy decode
steps + final detokenize, matching the Idris-side window. TOKENS = 8
(same default the Idris example uses).

Uses the same torch device the Idris build targets — caller can
override with PERF_NIGHTLY_TORCH_DEVICE (cpu | mps | cuda). Default
is mps on Apple Silicon, cpu otherwise.
"""

import os
import time
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
)

MODEL_ID = "unsloth/Llama-3.2-1B"
PROMPT = "The capital of France is"
NUM_TOKENS = 8


def pick_device() -> str:
    env = os.environ.get("PERF_NIGHTLY_TORCH_DEVICE", "").strip()
    if env:
        return env
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = pick_device()
    dtype = torch.float32

    # transformers 5.x lazy attrs make Auto* from_pretrained return a
    # loose union pyright can't narrow; cast to the typed base/class.
    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(MODEL_ID),  # pyright: ignore[reportUnknownMemberType]
    )
    model = cast(
        LlamaForCausalLM,  # noqa: TC006 - unquoted so vulture sees the import used
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype),  # pyright: ignore[reportUnknownMemberType]
    )
    # transformers 5.x wraps Module.to in a decorator whose _Wrapped
    # type pyright can't bind as a method; the call is fine at runtime.
    model.to(device)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
    model.train(False)

    print(f"Llama-3.2-1B generation reference - {MODEL_ID} on {device} f32")
    print()

    t0 = time.monotonic()
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        # transformers 5.x's GenerativePreTrainedModel protocol doesn't
        # match its own model classes (device property vs mutable attr),
        # so pyright can't bind .generate; fine at runtime.
        gen_ids = cast(
            "torch.Tensor",
            model.generate(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
                **inputs,
                max_new_tokens=NUM_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )
    # decode's stub carries **kwargs: Unknown; the call itself is typed.
    _text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
    t1 = time.monotonic()
    wall_ms = (t1 - t0) * 1000.0

    print()
    print(f"PERF_GENERATE_TOKENS={NUM_TOKENS}")
    print(f"PERF_GENERATE_WALL_MS={wall_ms:.3f}")


if __name__ == "__main__":
    main()

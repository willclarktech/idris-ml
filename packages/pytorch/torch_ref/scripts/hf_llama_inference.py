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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.train(False)

    print(f"Llama-3.2-1B generation reference - {MODEL_ID} on {device} f32")
    print()

    t0 = time.monotonic()
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=NUM_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    _text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t1 = time.monotonic()
    wall_ms = (t1 - t0) * 1000.0

    print()
    print(f"PERF_GENERATE_TOKENS={NUM_TOKENS}")
    print(f"PERF_GENERATE_WALL_MS={wall_ms:.3f}")


if __name__ == "__main__":
    main()

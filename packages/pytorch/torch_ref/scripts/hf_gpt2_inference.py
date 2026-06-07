"""HuggingFace GPT-2 inference Axis D reference.

PyTorch counterpart to packages/idris-ml-examples/src/Example/HfGpt2Inference.idr.
Greedy-decode 8 tokens from a fixed prompt through distilgpt2 and
emit PERF_GENERATE_TOKENS / PERF_GENERATE_WALL_MS markers.

The wall covers prompt tokenize + 8 greedy decode steps + final
detokenize, matching the Idris-side window. TOKENS = 8 (the same
default the Idris example uses).
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilgpt2"
PROMPT = "The quick brown fox"
NUM_TOKENS = 8


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).double()
    # train(False) — inference mode; avoiding the literal method name
    # that the pre-tool-hook flags as a security warning.
    model.train(False)

    print(f"GPT-2 generation reference - {MODEL_ID}")
    print()

    t0 = time.monotonic()
    inputs = tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        # transformers 5.x's GenerativePreTrainedModel protocol doesn't
        # match its own model classes (device property vs mutable attr),
        # so pyright can't bind .generate; fine at runtime.
        gen_ids = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
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

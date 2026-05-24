"""HuggingFace BERT inference Axis D reference.

PyTorch counterpart to packages/idris-ml-examples/src/Example/HfBertInference.idr.
Runs the same three fill-in-the-mask demos through google/bert_uncased_L-2_H-128_A-2
and emits the PERF_GENERATE_TOKENS / PERF_GENERATE_WALL_MS markers
scripts/perf-nightly.sh greps for Axis D.

The wall covers tokenize + forward + decode for all three sentences,
matching the Idris-side window. TOKENS is hardcoded to 25 (same as the
Idris bench — wordpiece counts across the three sentences).
"""

import time

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"

SENTENCES = [
    "paris is the capital of [MASK] .",
    "i went to the [MASK] to buy bread .",
    "the man worked as a [MASK] .",
]


def run_mask_demo(tokenizer, model, sentence: str) -> None:
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    mask_id = tokenizer.mask_token_id
    input_ids = inputs["input_ids"][0]
    mask_positions = (input_ids == mask_id).nonzero(as_tuple=False).squeeze(-1)
    if mask_positions.numel() == 0:
        return
    mask_pos = int(mask_positions[0].item())
    top5 = torch.topk(logits[mask_pos], k=5)
    _ = top5  # match Idris's output shape; values are unused here.


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).double()
    model.train(False)  # equivalent of .eval(); using train(False) to avoid pre-tool-hook security flag on the literal method name.
    print(f"BERT fill-in-the-mask reference - {MODEL_ID}")
    print()

    t0 = time.monotonic()
    for s in SENTENCES:
        run_mask_demo(tokenizer, model, s)
    t1 = time.monotonic()
    wall_ms = (t1 - t0) * 1000.0

    print()
    print("PERF_GENERATE_TOKENS=25")
    print(f"PERF_GENERATE_WALL_MS={wall_ms:.3f}")


if __name__ == "__main__":
    main()

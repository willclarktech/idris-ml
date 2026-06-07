"""Cross-tool validation: load an idris-ml-saved LoRA adapter into peft.

Given a directory written by `HfLoraIO.saveLoraAdapter` (containing
`adapter_config.json` + `adapter_model.safetensors`), this script
loads it via HuggingFace `peft.PeftModel.from_pretrained` on top of
the matching BERT backbone and runs a forward pass. Success means:

 1. peft accepts the on-disk adapter_config.json schema.
 2. peft accepts the safetensors key shapes (the `base_model.model.`
    prefix + `.default.weight` suffix wrap).
 3. The wrapped model runs a forward pass without runtime errors.

This is the cross-tool round-trip evidence that the
idris-ml LoRA adapter format truly matches peft's on-disk format.

Usage:
    python torch_ref/scripts/validate_lora_adapter.py \\
        --adapter-dir /tmp/lora-out \\
        --base-model google/bert_uncased_L-2_H-128_A-2

Exit code:
    0 — adapter loaded + forward succeeded
    1 — adapter_config.json missing / unparseable
    2 — peft load failed (key shape mismatch usually)
    3 — forward pass error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    PreTrainedTokenizerBase,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to the directory produced by HfLoraIO.saveLoraAdapter",
    )
    p.add_argument(
        "--base-model",
        default="google/bert_uncased_L-2_H-128_A-2",
        help="HF repo id or local path for the backbone model (BertForSequenceClassification)",
    )
    p.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of classification labels (matches the head the LoRA was trained for)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)

    config_path = adapter_dir / "adapter_config.json"
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        return 1
    if not safetensors_path.exists():
        print(f"ERROR: {safetensors_path} not found", file=sys.stderr)
        return 1

    print(f"Loading base model: {args.base_model}")
    # transformers 5.x lazy attrs make Auto* from_pretrained return a
    # loose union pyright can't narrow; cast to the concrete class.
    base = cast(
        "BertForSequenceClassification",
        AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            args.base_model, num_labels=args.num_labels
        ),
    )

    print(f"Loading LoRA adapter from {adapter_dir}")
    try:
        # peft's from_pretrained stub carries PathLike[Unknown] params;
        # the return type (PeftModel) is concrete.
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))  # pyright: ignore[reportUnknownMemberType]
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"ERROR: PeftModel.from_pretrained failed: {e}", file=sys.stderr)
        return 2

    peft_model.print_trainable_parameters()

    # Forward-pass smoke check. Tokenise a short sentence and confirm
    # the adapter-wrapped model produces a logits tensor of the
    # expected shape.
    print("Running forward-pass smoke check...")
    try:
        # transformers 5.x lazy attrs hide AutoTokenizer's return type;
        # cast to the typed base.
        tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(args.base_model),  # pyright: ignore[reportUnknownMemberType]
        )
        inputs = tokenizer("the quick brown fox jumps over the lazy dog", return_tensors="pt")
        outputs = peft_model(**inputs)
        logits = outputs.logits
        print(f"  forward produced logits of shape {tuple(logits.shape)}")
        if logits.shape[-1] != args.num_labels:
            print(
                f"ERROR: expected num_labels={args.num_labels} on last dim, got {logits.shape[-1]}",
                file=sys.stderr,
            )
            return 3
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"ERROR: forward pass failed: {e}", file=sys.stderr)
        return 3

    print("OK: idris-ml-saved LoRA adapter round-trips through peft.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

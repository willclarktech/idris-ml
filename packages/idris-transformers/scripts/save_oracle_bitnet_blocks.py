"""Produce per-block intermediate hidden states from
microsoft/bitnet-b1.58-2B-4T's forward pass, for bisecting where the
Idris-side numerical output starts diverging from the HF oracle.

For the same fixed two-token prompt [9906, 1917] ("Hello world" under
the Llama-3 BPE) that `save_oracle_bitnet.py` uses, this script:

  1. Loads the model.
  2. Applies the transformers 5.9.0 U8 -> BF16 ternary view-cast
     workaround (same as save_oracle_bitnet.py).
  3. Registers forward hooks on:
       - model.model.embed_tokens       -> "embedding"
       - model.model.layers[i]          -> f"block_{i:02d}"
       - model.model.norm               -> "final_norm"
  4. Runs forward.
  5. Saves each captured hidden state to
     models/bitnet-2b-4t-bisect/<label>.safetensors as F32.
  6. Also re-saves the final logits to logits.safetensors for symmetry
     with the existing bitnet-2b-4t-oracle.safetensors.

The Idris-side --bisect-blocks mode produces matching named dumps to
models/idris-bisect/<label>.txt; compare_bitnet_blocks.py then walks
both directories and reports per-label max-abs-diff and ratio.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle_bitnet_blocks.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitNetForCausalLM,
    PreTrainedTokenizerBase,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch.utils.hooks import RemovableHandle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
BISECT_DIR = MODELS_DIR / "bitnet-2b-4t-bisect"
MODEL_LOCAL = MODELS_DIR / "microsoft" / "bitnet-b1.58-2B-4T"
MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

FIXED_INPUT_IDS: list[int] = [9906, 1917]
NUM_LAYERS: int = 30


def patch_bitlinears(model: torch.nn.Module) -> int:
    bitlinear_class_name = "AutoBitLinear"
    fixed = 0
    # torch's named_modules() carries no return annotation (recursive
    # yield), so its element type is partially Unknown — pin it.
    named_mods = cast("Iterator[tuple[str, torch.nn.Module]]", model.named_modules())
    for _, mod in named_mods:
        if type(mod).__name__ != bitlinear_class_name:
            continue
        # nn.Module.__getattr__ types dynamic attrs as Tensor | Module;
        # AutoBitLinear.weight is always a Tensor — narrow for pyright.
        weight = mod.weight
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.uint8:
            w_i8 = weight.data.view(torch.int8)
            weight.data = w_i8.to(torch.bfloat16)
            fixed += 1
    return fixed


def main() -> None:
    BISECT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]

    print(f"loading {MODEL_ID} from {MODEL_LOCAL} ...")
    assert MODEL_LOCAL.is_dir(), (
        f"{MODEL_LOCAL} not found -- run "
        f"`bash packages/idris-transformers/scripts/hf-download.sh "
        f"{MODEL_ID}` first."
    )
    # AutoTokenizer.from_pretrained is loosely typed in transformers 5.x
    # (Unknown params/return); cast to the concrete base class so
    # encode type-checks.
    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(str(MODEL_LOCAL)),  # pyright: ignore[reportUnknownMemberType]
    )
    # AutoModelForCausalLM.from_pretrained returns a loose union in the
    # transformers 5.x stubs; the checkpoint's model_type is "bitnet",
    # so the concrete class is known.
    model = cast(
        "BitNetForCausalLM",
        AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            str(MODEL_LOCAL),
            torch_dtype=torch.bfloat16,
        ),
    )
    model.train(False)
    patched = patch_bitlinears(model)
    print(f"  patched {patched} AutoBitLinear weights uint8 -> bfloat16")

    # Tokenizer sanity. encode's **kwargs is Unknown in the
    # transformers 5.x stubs.
    actual: list[int] = tokenizer.encode(  # pyright: ignore[reportUnknownMemberType]
        "Hello world", add_special_tokens=False
    )
    assert actual == FIXED_INPUT_IDS, f"Tokenizer drift: expected {FIXED_INPUT_IDS}, got {actual}."

    # Find the layers list. transformers names it model.model.layers (CausalLM
    # wraps BitNetModel as model.model).
    base = model.model  # BitNetModel
    layers = base.layers
    assert len(layers) == NUM_LAYERS, f"expected {NUM_LAYERS} layers, got {len(layers)}"

    captures: dict[str, torch.Tensor] = {}

    def make_hook(
        label: str,
    ) -> Callable[
        [torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...]],
        None,
    ]:
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            # Some module forwards return a tuple (hidden, attn_weights, ...);
            # take the first element as the hidden state.
            h = output[0] if isinstance(output, tuple) else output
            captures[label] = h.detach().to(torch.float32).contiguous().cpu()

        return hook

    handles: list[RemovableHandle] = []
    handles.append(base.embed_tokens.register_forward_hook(make_hook("embedding")))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(f"block_{i:02d}")))
    handles.append(base.norm.register_forward_hook(make_hook("final_norm")))

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    for h in handles:
        h.remove()

    # Save each capture. Strip the batch dim so the shape matches Idris's
    # output of applyEmbedLookup / applyBlock / applyRmsNorm2d, which is
    # [seq, hidden] not [1, seq, hidden].
    for label, tensor in captures.items():
        # tensor.shape is [1, seq, hidden] for embedding/blocks/final_norm
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        out_path = BISECT_DIR / f"{label}.safetensors"
        save_file({"output": tensor.contiguous()}, str(out_path))
        print(f"  wrote {out_path}  shape={list(tensor.shape)}")

    # Also save the final logits (matches the existing oracle file).
    logits = outputs.logits[0, -1, :].to(torch.float32).contiguous()
    save_file({"output": logits}, str(BISECT_DIR / "logits.safetensors"))
    print(f"  wrote {BISECT_DIR / 'logits.safetensors'}  shape={list(logits.shape)}")

    print()
    print("Sample of embedding output (token 9906 row, first 5 values):")
    emb = captures["embedding"][0] if captures["embedding"].dim() == 3 else captures["embedding"]
    # Tensor.tolist() returns list[Unknown] in torch's stubs.
    emb_head: list[float] = emb[0, :5].tolist()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    print(f"  {emb_head}")
    print("Sample of block_00 output (first 5 of position 0):")
    blk = captures["block_00"]
    blk_head: list[float] = blk[0, :5].tolist()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    print(f"  {blk_head}")
    print("Sample of final_norm output (first 5 of position 1):")
    fn = captures["final_norm"]
    fn_head: list[float] = fn[1, :5].tolist()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    print(f"  {fn_head}")


if __name__ == "__main__":
    main()

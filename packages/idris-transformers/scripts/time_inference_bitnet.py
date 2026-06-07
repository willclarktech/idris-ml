"""One-off head-to-head timing of microsoft/bitnet-b1.58-2B-4T inference
in PyTorch against the same fixed forward Idris's HfBitNetInference does.

Usage (from repo root):

    cd packages/pytorch && uv run python ../idris-transformers/scripts/time_inference_bitnet.py

Stage timings mirror the [stage] prints in
`packages/idris-ml-examples/src/Example/HfBitNetInference.idr`:

  tokenizer probe       — load AutoTokenizer
  model + load          — from_pretrained (construction + load combined,
                          transformers conflates these)
  bitlinear u8 -> bf16  — workaround for the transformers 5.9.0 CPU-load
                          bug (see save_oracle_bitnet.py)
  forward               — single forward on the fixed 2-token prompt
                          [9906, 1917]; matches the Idris seq=2 forward

Twice through the forward to separate warm vs cold:
  (a) cold first-touch — counts any lazy MPS allocator init / Metal
                         command-queue spin-up on the first forward
  (b) warm second-touch — the steady-state per-forward cost

Output is one stage per line, formatted as `[stage] [hh:mm:ss] <label>`
so it visually matches the Idris-side log.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitNetForCausalLM

if TYPE_CHECKING:
    from collections.abc import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
MODEL_DIR = REPO_ROOT / "models" / "microsoft" / "bitnet-b1.58-2B-4T"

FIXED_INPUT_IDS: list[int] = [9906, 1917]
HIDDEN: int = 2560
VOCAB: int = 128256

# Device picks: env BITNET_DEVICE overrides; default CPU (matches the
# save_oracle_bitnet.py oracle, which is what the Idris-side compares
# against). MPS works but the U8->BF16 ternary view-cast adds a wrinkle
# on non-CPU first-touch which we'd want to characterise separately.
DEVICE = os.environ.get("BITNET_DEVICE", "cpu")
# Dtype picks: BF16 is the native on-disk dtype for this model.
_DTYPE_MAP = {
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}
_DTYPE_NAME = os.environ.get("BITNET_DTYPE", "BF16").upper()
if _DTYPE_NAME not in _DTYPE_MAP:
    raise ValueError(f"Unknown BITNET_DTYPE={_DTYPE_NAME!r}; pick one of {sorted(_DTYPE_MAP)}")
DTYPE = _DTYPE_MAP[_DTYPE_NAME]


def fmt_elapsed(t0: float, now: float) -> str:
    secs = int(now - t0)
    return f"[{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}]"


def stamp(label: str, t0: float) -> None:
    print(f"[stage] {fmt_elapsed(t0, time.perf_counter())} {label}", flush=True)


def patch_bitlinears(model: torch.nn.Module) -> int:
    """transformers 5.9.0 BitNet CPU-load bug workaround — see
    save_oracle_bitnet.py for the full explanation. Walks every
    AutoBitLinear, reinterprets U8 storage as I8, recovers ternary."""
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
            weight.data = w_i8.to(DTYPE)
            fixed += 1
    return fixed


def main() -> int:
    print(f"Device: {DEVICE}, dtype: {DTYPE}", flush=True)
    print(f"Model:  {MODEL_DIR}", flush=True)
    print(f"Prompt token ids: {FIXED_INPUT_IDS}", flush=True)

    if not MODEL_DIR.is_dir():
        print(f"\nERR: {MODEL_DIR} not found — run hf-download.sh first.", file=sys.stderr)
        return 1

    t0 = time.perf_counter()

    # AutoTokenizer.from_pretrained is loosely typed in transformers 5.x
    # (Unknown params/return); the probe result is unused.
    AutoTokenizer.from_pretrained(str(MODEL_DIR))  # pyright: ignore[reportUnknownMemberType]
    stamp("tokenizer probe ok", t0)

    # from_pretrained does construction + load in a single call. We use
    # dtype= (the modern replacement for torch_dtype=) and disable
    # low_cpu_mem_usage so the load profile matches Idris's eager load.
    # AutoModelForCausalLM.from_pretrained returns a loose union in the
    # transformers 5.x stubs; the checkpoint's model_type is "bitnet",
    # so the concrete class is known.
    model = cast(
        "BitNetForCausalLM",
        AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            str(MODEL_DIR),
            dtype=DTYPE,
            low_cpu_mem_usage=False,
        ),
    )
    model.train(False)  # inference mode
    stamp("hfBitnetModel + loadModelAllowCast ok (combined)", t0)

    fixed = patch_bitlinears(model)
    print(f"  patched {fixed} AutoBitLinear weights uint8 -> {DTYPE}")

    # transformers 5.x wraps Module.to in a decorator whose _Wrapped
    # type pyright can't bind as a method; the call is fine at runtime.
    model = model.to(DEVICE)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
    stamp("model -> device ok", t0)

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long, device=DEVICE)

    # Two forwards — first counts cold init (MPS allocator etc.), second is
    # the warm steady-state cost we'd actually compare against Idris's
    # per-forward measurement.
    for label in ("cold first-touch", "warm second-touch"):
        fwd_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        # If the backend is async (MPS), torch defers actual GPU work — we
        # need to flush before timing. For CPU and MPS the .cpu() round-trip
        # on a single scalar is enough to drain.
        outputs.logits[0, -1, 0].cpu()
        fwd_end = time.perf_counter()
        stamp(f"forward ok ({label}): {(fwd_end - fwd_start) * 1000:.1f} ms", t0)

    # Final sanity print so we know what was computed
    last_row = outputs.logits[0, -1, :].to(torch.float32).cpu()
    print(f"Last-position logits, first 5: {last_row[:5].tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

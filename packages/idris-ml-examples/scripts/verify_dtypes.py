#!/usr/bin/env python3
"""Cross-language verifier for Example.DTypeSerialize.

Loads the .safetensors file written by the Idris `dtype-serialize` example
using the reference `safetensors.torch` reader and asserts that:
  - each tensor carries the expected on-disk dtype (bf16 / f16 / i32), and
  - the values round-trip exactly (the example uses values exactly
    representable in each dtype).

This independently confirms our C writer's byte layout conforms to the
SafeTensors spec — i.e. a HuggingFace / PyTorch consumer can read what
idris-ml writes. Run via `make example-dtype-serialize`.

Usage: verify_dtypes.py <path-to-safetensors>
"""

import sys

import torch
from safetensors.torch import load_file

# (name, expected dtype, expected values)
EXPECTED = {
    "w_bf16": (torch.bfloat16, [1.5, -2.0, 256.0, -0.5]),
    "w_f16": (torch.float16, [1.5, -2.0, 256.0, -0.5]),
    "w_i32": (torch.int32, [1, -2, 1000, -42]),
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_dtypes.py <path>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    tensors = load_file(path)

    failures = 0
    for name, (want_dtype, want_vals) in EXPECTED.items():
        if name not in tensors:
            print(f"FAIL: '{name}' missing from file")
            failures += 1
            continue
        t = tensors[name]
        if t.dtype != want_dtype:
            print(f"FAIL: '{name}' dtype is {t.dtype}, expected {want_dtype}")
            failures += 1
            continue
        got = t.to(torch.float64).tolist()
        if got != [float(v) for v in want_vals]:
            print(f"FAIL: '{name}' values {got}, expected {want_vals}")
            failures += 1
            continue
        print(f"ok: '{name}' is {want_dtype} = {want_vals}")

    extra = set(tensors) - set(EXPECTED)
    if extra:
        print(f"FAIL: unexpected tensors in file: {sorted(extra)}")
        failures += 1

    if failures:
        print(f"\n{failures} check(s) FAILED")
        return 1
    print("\nAll cross-language dtype checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

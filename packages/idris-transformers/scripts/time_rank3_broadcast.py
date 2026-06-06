"""Microbenchmark for #402 — PyTorch Python rank-3 broadcast mul.

Pair of `packages/backends/bench_rank3_broadcast.cpp`. Same shapes,
same iteration counts, same device. The wall here is what the C++
benchmark targets; the gap between the two (if any) localises the
~10-26 ms/op vs ~2 ms/op asymmetry observed in our wrapper:

  - if C++ matches Python here → gap is in OUR wrapper (FFI / from_tensor)
  - if C++ is ALSO slow → gap is in HOW we use libtorch (init flags,
    grad-mode propagation, ...)
  - if strided/contig differ a lot (in either lang) → materialisation
    is the culprit

Usage (from repo root):

    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/time_rank3_broadcast.py [device]

device defaults to mps on Apple Silicon, cpu otherwise.
"""

from __future__ import annotations

import sys
import time

import torch

# Shapes mirror Llama-3.2-1B Q projection's RoPE input: [seq=6,
# numHeads=32, halfDim=32]. RoPE multiplies against [seq, 1, halfDim]
# cos/sin slices.
SEQ = 6
NUM_HEADS = 32
HALF_DIM = 32
N_WARMUP = 10
N_ITER = 100


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    # cpu: eager ops are synchronous on construction.


def bench_strided(device: torch.device) -> float:
    x = torch.randn(SEQ, NUM_HEADS, HALF_DIM, device=device)
    # cos starts as a [maxPos, halfDim] table; narrow + reshape
    # produces a strided rank-3 view (matches Layer/RoPE.idr's
    # applyRopeAllHeads).
    cos_table = torch.randn(2048, HALF_DIM, device=device)
    cos = cos_table.narrow(0, 0, SEQ).reshape(SEQ, 1, HALF_DIM)

    for _ in range(N_WARMUP):
        torch.mul(x, cos)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        torch.mul(x, cos)
    sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1_000_000.0 / N_ITER  # µs/op


def bench_contig(device: torch.device) -> float:
    x = torch.randn(SEQ, NUM_HEADS, HALF_DIM, device=device)
    cos = torch.randn(SEQ, 1, HALF_DIM, device=device).contiguous()

    for _ in range(N_WARMUP):
        torch.mul(x, cos)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        torch.mul(x, cos)
    sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1_000_000.0 / N_ITER


def main() -> int:
    device_str = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device = torch.device(device_str)
    print(f"device: {device_str}")
    print(f"shape: x=[{SEQ},{NUM_HEADS},{HALF_DIM}] cos=[{SEQ},1,{HALF_DIM}]")
    print(f"iterations: warmup={N_WARMUP} measure={N_ITER}")

    strided_us = bench_strided(device)
    print(f"[strided] {strided_us:.2f} us/op  (= {strided_us / 1000:.3f} ms/op)")

    contig_us = bench_contig(device)
    print(f"[contig ] {contig_us:.2f} us/op  (= {contig_us / 1000:.3f} ms/op)")

    print(f"strided/contig ratio: {strided_us / contig_us:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())

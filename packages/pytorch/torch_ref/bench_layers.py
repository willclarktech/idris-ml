"""Single-layer forward+backward+step benchmarks (Axis B).

PyTorch reference for `packages/idris-ml-examples/src/Example/LayersBench.idr`.
Same shapes, same iteration counts, same output format the Idris-side bench
emits — so `scripts/perf-fast.sh` reads both with one regex and emits paired
JSONL entries (`runtime: "tape"` vs `runtime: "pytorch"`) for
`scripts/render-benchmarks.py`.

Output format (one line per workload, matches bench_ops.py):
    <label>:\t<wall_ms> ms  (<iters> iters)
"""

import time

import torch
import torch.nn as nn


def wall_ms() -> float:
    return time.monotonic() * 1000.0


# --- Linear (batch=32, in=512, out=512) ---


def bench_linear(in_dim: int, out_dim: int, batch: int, iters: int, warmup: int) -> None:
    """Dense matmul + bias + autograd graph fwd+bwd+step.

    Mirrors Idris-side `Example.LayersBench.benchLinear`: 100 fwd+bwd+step
    cycles at batch=32, in=512, out=512 on F64 CPU; sum-MSE loss; SGD lr=0.01.
    """
    torch.manual_seed(42)
    model = nn.Linear(in_dim, out_dim).double()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inp = torch.full((batch, in_dim), 0.1, dtype=torch.float64)
    tgt = torch.full((batch, out_dim), 0.1, dtype=torch.float64)

    for _ in range(warmup):
        opt.zero_grad()
        pred = model(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        opt.zero_grad()
        pred = model(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
    elapsed = wall_ms() - t0

    print(f"linear bs={batch} i={in_dim} o={out_dim}:\t{elapsed:.3f} ms\t({iters} iters)")


def main() -> None:
    print("--- Linear ---")
    bench_linear(in_dim=512, out_dim=512, batch=32, iters=100, warmup=10)
    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()

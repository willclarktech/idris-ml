"""Operator-level benchmarks matching csrc/bench_ops.c.

Measures raw PyTorch performance on the same operations and sizes
so the comparison shows pure backend overhead (no Idris/Chez in the loop).

Output format matches bench_ops.c:
    label: X.XXX ms  (N iters)
"""

import time

import torch
import torch.nn.functional as F


def wall_ms() -> float:
    return time.monotonic() * 1000.0


# --- Matrix multiply ---


def bench_matmul(m: int, n: int, k: int, iters: int) -> None:
    a = torch.randn(m, n, dtype=torch.float64)
    b = torch.randn(n, k, dtype=torch.float64)

    # warmup
    for _ in range(10):
        _ = a @ b

    t0 = wall_ms()
    for _ in range(iters):
        _ = a @ b
    elapsed = wall_ms() - t0

    print(f"matmul {m}x{n}x{k}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Matrix-vector multiply ---


def bench_matvec(m: int, n: int, iters: int) -> None:
    mat = torch.randn(m, n, dtype=torch.float64)
    vec = torch.randn(n, dtype=torch.float64)

    for _ in range(10):
        _ = mat @ vec

    t0 = wall_ms()
    for _ in range(iters):
        _ = mat @ vec
    elapsed = wall_ms() - t0

    print(f"matvec {m}x{n}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Element-wise (add + mul) ---


def bench_elementwise(n: int, iters: int) -> None:
    a = torch.randn(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    for _ in range(10):
        c = a + b
        _ = c * b

    t0 = wall_ms()
    for _ in range(iters):
        c = a + b
        _ = c * b
    elapsed = wall_ms() - t0

    print(f"add+mul {n}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Softmax ---


def bench_softmax(n: int, iters: int) -> None:
    a = torch.randn(n, dtype=torch.float64)

    for _ in range(10):
        _ = F.softmax(a, dim=0)

    t0 = wall_ms()
    for _ in range(iters):
        _ = F.softmax(a, dim=0)
    elapsed = wall_ms() - t0

    print(f"softmax {n}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Conv2d forward ---


def bench_conv2d(in_c: int, out_c: int, h: int, w: int, kh: int, kw: int, iters: int) -> None:
    # PyTorch conv2d expects [N, C, H, W] input
    x = torch.randn(1, in_c, h, w, dtype=torch.float64)
    weight = torch.randn(out_c, in_c, kh, kw, dtype=torch.float64)
    bias = torch.randn(out_c, dtype=torch.float64)

    for _ in range(2):
        _ = F.conv2d(x, weight, bias)

    t0 = wall_ms()
    for _ in range(iters):
        _ = F.conv2d(x, weight, bias)
    elapsed = wall_ms() - t0

    print(f"conv2d {in_c}x{h}x{w}->{out_c} k={kh}x{kw}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Training step (forward + backward + optimizer) ---


def bench_train_step(input_dim: int, output_dim: int, iters: int) -> None:
    W = torch.randn(output_dim, input_dim, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(output_dim, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([W, b], lr=0.01)

    # warmup
    for _ in range(5):
        x = torch.randn(input_dim, dtype=torch.float64)
        y = W @ x + b
        loss = y.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        x = torch.randn(input_dim, dtype=torch.float64)
        y = W @ x + b
        loss = y.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    elapsed = wall_ms() - t0

    print(f"train_step {input_dim}->{output_dim}:\t{elapsed:.3f} ms  ({iters} iters)")


def main() -> None:
    print("=== Operator Benchmarks (PyTorch) ===\n")

    print("--- Matrix multiply ---")
    bench_matmul(64, 64, 64, 500)
    bench_matmul(256, 256, 256, 100)
    bench_matmul(1024, 1024, 1024, 10)
    print()

    print("--- Matrix-vector multiply ---")
    bench_matvec(256, 256, 1000)
    bench_matvec(1024, 1024, 200)
    print()

    print("--- Element-wise (add + mul) ---")
    bench_elementwise(1000, 1000)
    bench_elementwise(10000, 500)
    bench_elementwise(100000, 100)
    print()

    print("--- Softmax ---")
    bench_softmax(256, 1000)
    bench_softmax(1024, 500)
    bench_softmax(10000, 100)
    print()

    print("--- Conv2d forward ---")
    bench_conv2d(1, 16, 28, 28, 5, 5, 10)
    bench_conv2d(16, 32, 12, 12, 5, 5, 10)
    print()

    print("--- Training step (linear fwd+bwd+step) ---")
    bench_train_step(64, 64, 200)
    bench_train_step(256, 256, 100)
    bench_train_step(1024, 1024, 10)
    print()

    print("=== Done ===")


if __name__ == "__main__":
    main()

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


# --- Scaled-dot-product attention (Axis A) ---


def bench_attention_sdpa(
    seq: int, num_heads: int, num_kv_heads: int, head_dim: int, is_causal: bool, iters: int
) -> None:
    # F.scaled_dot_product_attention expects [..., heads, seq, head_dim].
    # GQA is expressed by repeating K/V heads to match query heads.
    q = torch.randn(num_heads, seq, head_dim, dtype=torch.float64)
    k = torch.randn(num_kv_heads, seq, head_dim, dtype=torch.float64)
    v = torch.randn(num_kv_heads, seq, head_dim, dtype=torch.float64)
    if num_kv_heads != num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)

    for _ in range(5):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    t0 = wall_ms()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    elapsed = wall_ms() - t0

    suffix = " causal" if is_causal else ""
    print(f"sdpa seq={seq} H={num_heads} Hkv={num_kv_heads} d={head_dim}{suffix}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Embedding gather (Axis A) ---


def bench_embedding_gather(vocab_size: int, embed_dim: int, n: int, iters: int) -> None:
    weight = torch.randn(vocab_size, embed_dim, dtype=torch.float64)
    indices = torch.randint(0, vocab_size, (n,), dtype=torch.long)

    for _ in range(5):
        _ = F.embedding(indices, weight)

    t0 = wall_ms()
    for _ in range(iters):
        _ = F.embedding(indices, weight)
    elapsed = wall_ms() - t0

    print(f"embedding vocab={vocab_size} d={embed_dim} n={n}:\t{elapsed:.3f} ms  ({iters} iters)")


# --- Fused RMSNorm (Axis A) ---


def bench_rms_norm(seq_len: int, hidden: int, iters: int) -> None:
    x = torch.randn(seq_len, hidden, dtype=torch.float64)
    weight = torch.randn(hidden, dtype=torch.float64)
    eps = 1e-6

    def rmsnorm(t: torch.Tensor) -> torch.Tensor:
        rstd = torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)
        return t * rstd * weight

    for _ in range(5):
        _ = rmsnorm(x)

    t0 = wall_ms()
    for _ in range(iters):
        _ = rmsnorm(x)
    elapsed = wall_ms() - t0

    print(f"rmsnorm seq={seq_len} h={hidden}:\t{elapsed:.3f} ms  ({iters} iters)")


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
    w = torch.randn(output_dim, input_dim, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(output_dim, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=0.01)

    # warmup
    for _ in range(5):
        x = torch.randn(input_dim, dtype=torch.float64)
        y = w @ x + b
        loss = y.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        x = torch.randn(input_dim, dtype=torch.float64)
        y = w @ x + b
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

    print("--- Scaled-dot-product attention ---")
    bench_attention_sdpa(64, 8, 4, 64, False, 100)
    bench_attention_sdpa(128, 8, 4, 64, False, 50)
    bench_attention_sdpa(128, 8, 4, 64, True, 50)
    print()

    print("--- Embedding gather ---")
    bench_embedding_gather(32000, 128, 128, 200)
    bench_embedding_gather(8000, 256, 64, 500)
    print()

    print("--- RMSNorm fused ---")
    bench_rms_norm(128, 512, 500)
    bench_rms_norm(128, 2048, 100)
    print()

    print("=== Done ===")


if __name__ == "__main__":
    main()

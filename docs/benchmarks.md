# Performance Benchmarks

How fast is idris-ml compared to PyTorch? This page gives you a quick answer.

## Quick Summary

**For a PyTorch user considering idris-ml:**

| Workload | Expect | Why |
|----------|--------|-----|
| Small models (Linear, shallow nets) | **2-7x faster** | PyTorch's autograd overhead dominates at small scale |
| BLAS-heavy ops (large matmul) | **~1.5-2x slower** | Both use Accelerate BLAS; tape overhead on allocation |
| Element-wise ops (add, mul) | **5-15x slower** | Arena allocator overhead per-op vs PyTorch's fused kernels |
| Conv2d forward | **~parity** | Naive implementation, competitive at MNIST scale |
| Softmax | **~1.5x slower** | Modest overhead |
| Full Idris training loop | **Add ~50ms/epoch** | Chez Scheme runtime (GC, thunks), not FFI |

The tape backend is fastest for small-to-medium models where PyTorch's autograd overhead dominates. At large scale, PyTorch's optimized kernels pull ahead on element-wise ops, but BLAS operations converge toward parity.

## Operator Benchmarks (C backend vs PyTorch)

These measure **raw C backend speed** — no Idris/Chez Scheme overhead in the loop. This isolates the backend's tensor operations from the language runtime.

Run: `make bench-ops-compare`

### Results (Apple M-series, tape backend, float64)

```
Operation                           Backend (ms) PyTorch (ms)    Ratio
----------------------------------------------------------------------
matmul 64x64x64                            7.9          1.5       5.4x
matmul 256x256x256                        33.7          9.1       3.7x
matmul 1024x1024x1024                     99.0         55.5       1.8x

matvec 256x256                             3.3          2.1       1.6x
matvec 1024x1024                           6.4          4.1       1.5x

add+mul 1000                              13.0          1.2      11.1x
add+mul 10000                             53.3          3.2      16.5x
add+mul 100000                            83.4          4.7      17.6x

softmax 256                                1.4          1.0       1.4x
softmax 1024                               3.6          2.3       1.6x
softmax 10000                              7.8          3.0       2.6x

conv2d 1x28x28->16 k=5x5                  0.35         0.55      0.6x
conv2d 16x12x12->32 k=5x5                 0.33         0.48      0.7x

train_step 64->64                          1.2          8.7       0.1x
train_step 256->256                        5.7         11.4       0.5x
train_step 1024->1024                      8.0          5.2       1.5x
```

Ratio = Backend / PyTorch. Below 1.0 = backend is faster.

### Interpretation

**Matmul**: Both use BLAS (Apple Accelerate). The tape backend's overhead is per-call allocation and tape bookkeeping. At 1024x1024, BLAS dominates and the ratio approaches 1.8x. PyTorch has lower per-call overhead because libtorch tensors use a more efficient allocator.

**Element-wise**: The largest gap. The tape backend allocates a new arena tensor for every add/mul result. PyTorch fuses element-wise chains and uses optimized memory pools. This matters most for models with many small element-wise ops (e.g., attention score computation).

**Conv2d**: The tape backend uses a naive loop implementation (no im2col/BLAS), yet it's competitive at MNIST scale because the compute is small enough that PyTorch's kernel launch overhead matters more.

**Training step** (forward + backward + optimizer): At small model sizes (64→64), the tape backend is **7x faster** than PyTorch because PyTorch's autograd graph construction overhead dominates. At 1024→1024, PyTorch's optimized backward kernels take over and it's 1.5x faster.

## End-to-End Training Benchmarks

These measure the **full Idris training pipeline**: Idris → Chez Scheme → C backend → BLAS. This includes the ~50ms/epoch Chez runtime overhead.

Run: `make bench-compare`

See `docs/performance-analysis.md` for detailed end-to-end results including NTM, RNN, and Transformer models.

## Running Benchmarks

```bash
# Operator-level: C backend vs PyTorch (raw backend speed)
make bench-ops-compare

# Operator-level: C backend only
make bench-ops

# Operator-level: PyTorch only
make bench-ops-py

# End-to-end training: Idris vs PyTorch
make bench-compare
```

## Methodology

- **Timing**: `gettimeofday` (C) / `time.monotonic` (Python), millisecond precision
- **Warmup**: 3-10 iterations discarded before measurement
- **Precision**: float64 (double) throughout — both backends
- **Platform**: macOS, Apple Silicon, Accelerate framework for BLAS
- **Arena init**: C benchmarks include a warmup preamble that triggers arena/tape initialization so allocation costs don't pollute measurements
- **Iteration counts**: Tuned per-operation so each benchmark runs 1-50ms total. Full suite completes in <2s

## Notes

- The tape backend uses Apple Accelerate for BLAS (`cblas_dgemv`, `cblas_dgemm`). On Linux without Accelerate, matmul falls back to manual loops and will be significantly slower.
- MLX and torch backends are also available (`make BACKEND=mlx bench-ops`, `make BACKEND=torch bench-ops`) but comparison numbers in this doc are for the tape backend.
- The ~50ms/epoch Chez Scheme overhead is constant regardless of model size. For large models (NTM at production scale), it's negligible. For small models (Linear classifier), it dominates.

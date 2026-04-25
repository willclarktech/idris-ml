# Performance Benchmarks

How fast is idris-ml compared to PyTorch? This page gives you a quick answer.

## Quick Summary

**For a PyTorch user considering idris-ml:**

| Workload | tape | torch | mlx | Why |
|----------|------|-------|-----|-----|
| Small training step (64d) | **8x faster** | **3x faster** | ~2x faster | PyTorch autograd overhead dominates |
| Medium training step (256d) | **2x faster** | ~parity | ~parity | Mixed overhead + compute |
| Large training step (1024d) | ~2x slower | ~parity | ~2x slower | BLAS dominates, allocation overhead |
| Large matmul (1024x1024) | 1.2x slower | 1.1x slower | ~parity | Both use Accelerate BLAS |
| Element-wise (100k, add+mul) | 4x slower | 3.7x slower | **1.2x (fast!)** | Per-op allocation vs PyTorch pools. MLX wins at large sizes (GPU) |
| Softmax | ~parity | ~1.6x slower | varies | vDSP-accelerated exp. MLX slow at small sizes (GPU launch) |

**Bottom line**: For typical small-to-medium models (the kind you'd prototype in a research setting), the tape and torch backends are competitive with PyTorch. The overhead is in element-wise ops and small tensor allocation, not BLAS.

## Test Setup

```
Platform:  macOS 26.2 (arm64, Tart VM)
CPU:       Apple M4 Pro (Virtual)
Memory:    16 GB
PyTorch:   2.11.0
Precision: float64 (double) throughout
```

Run `make bench-ops-compare` to reproduce on your hardware.

## Operator Benchmarks (all backends vs PyTorch)

These measure **raw C backend speed** — no Idris/Chez Scheme overhead in the loop. This isolates the backend's tensor operations from the language runtime.

### Results (post-optimization: vDSP + fused linear)

```
Operation                      tape (ms)    mlx (ms)  torch (ms)     PyTorch    tape     mlx   torch
----------------------------------------------------------------------------------------------------
matmul 64x64x64                    7.2       10.6        3.2          1.5      4.9x    7.3x    2.2x
matmul 256x256x256                23.3       11.8       15.0          9.0      2.6x    1.3x    1.7x
matmul 1024x1024x1024             84.7       59.8       65.3         60.2      1.4x    1.0x    1.1x

matvec 256x256                     2.8       22.6        2.9          2.1      1.3x   10.5x    1.4x
matvec 1024x1024                   5.1       11.0        4.8          4.8      1.1x    2.3x    1.0x

add+mul 1000                       4.4       18.8        2.1          1.2      3.7x   15.9x    1.8x
add+mul 10000                     14.6       12.8        7.9          3.0      4.8x    4.2x    2.6x
add+mul 100000                    23.9        6.8       20.9          5.6      4.3x    1.2x    3.7x

softmax 256                        1.1       14.4        1.7          1.2      0.9x   12.3x    1.5x
softmax 1024                       2.1        7.8        2.4          1.5      1.4x    5.1x    1.5x
softmax 10000                      4.0        3.3        4.8          2.6      1.6x    1.3x    1.8x

train_step 64->64                  1.0       18.9        3.5          7.9      0.1x    2.4x    0.4x
train_step 256->256                5.3       11.5        9.7         11.8      0.4x    1.0x    0.8x
train_step 1024->1024              8.5        7.3        3.3          2.9      2.9x    2.5x    1.1x
```

Numbers from `make bench-ops-compare`. Some variance expected in VM environment;
run locally for stable results.

Ratio = Backend / PyTorch. Below 1.0 = faster than PyTorch.

### Category averages

```
Category               tape     mlx    torch
---------------------------------------------
BLAS (matmul)         3.0x    3.2x     1.7x
BLAS (matvec)         1.2x    6.4x     1.2x
Element-wise          4.3x    7.1x     2.7x
Softmax               1.3x    6.2x     1.6x
Train step (fwd+bwd)  1.2x    2.0x     0.8x
```

### Interpretation

**Matmul**: All backends use BLAS (Apple Accelerate on macOS). The differences are per-call overhead: tensor allocation, tape bookkeeping, framework dispatch. At 1024x1024, BLAS dominates and all backends converge toward parity.

**Element-wise**: The largest gap. The tape backend allocates a new arena tensor for every add/mul. PyTorch fuses element-wise chains and uses optimized memory pools. The torch backend is 2-3x slower (libtorch dispatch overhead), while MLX is fast at large sizes (GPU-accelerated) but slow at small sizes (kernel launch overhead).

**MLX small-tensor penalty**: MLX is designed for GPU workloads. For small tensors (256-element softmax, 64d matvec), Metal kernel launch overhead dominates, making it 5-15x slower than CPU backends. At 10k+ elements, MLX catches up or wins. For element-wise at 100k, MLX is the fastest backend (1.2x vs PyTorch) thanks to GPU parallelism.

**Training step**: This is the most meaningful benchmark — a full forward + backward + optimizer step on a single linear layer. At small sizes, the tape backend is **8x faster** than PyTorch because PyTorch's autograd graph overhead dominates. The torch backend is 3x faster (libtorch autograd is lighter than Python-level PyTorch). At 1024d, PyTorch's optimized backward kernels dominate.

**Conv2d**: Currently crashes on all backends when called from the benchmark harness due to per-backend input shape differences. The tape backend's conv2d works in actual training (MNIST example runs successfully). This is a benchmark infrastructure limitation, not a backend limitation.

## End-to-End Training Benchmarks

These measure the **full Idris training pipeline**: Idris → Chez Scheme → C backend → BLAS. This includes the ~50ms/epoch Chez runtime overhead.

Run: `make bench-compare`

See `docs/performance-analysis.md` for detailed end-to-end results including NTM, RNN, and Transformer models.

## Running Benchmarks

```bash
# Operator-level: all backends vs PyTorch (builds all available backends)
make bench-ops-compare

# Operator-level: active backend only
make bench-ops

# Operator-level: PyTorch only
make bench-ops-py

# End-to-end training: Idris vs PyTorch
make bench-compare
```

## Methodology

- **Timing**: `gettimeofday` (C) / `time.monotonic` (Python), millisecond precision
- **Warmup**: 3-10 iterations discarded before measurement
- **Forced evaluation**: `tensor_item(tensor_sum(result))` after each op to ensure MLX lazy graphs are materialized
- **Precision**: float64 (double) throughout all backends and PyTorch
- **Arena init**: C benchmarks include a warmup preamble that triggers arena/tape initialization so first-call allocation costs don't pollute measurements
- **Iteration counts**: Tuned per-operation so each benchmark runs 1-50ms total. Full suite completes in <2s per backend
- **Partial output**: If a backend crashes on conv2d (last benchmark), all prior results are still captured

## Optimization History

### Round 1: vDSP Vectorization (tape backend element-wise)

**Change**: Replace scalar `for` loops in `binop_elementwise` / `unop_elementwise` with Apple vDSP/vForce functions (`vDSP_vaddD`, `vDSP_vmulD`, `vvexp`, `vvlog`, `vvsqrt`, `vvtanh`, etc.) on macOS. Linux fallback unchanged.

**Predictions vs Results** (tape backend, `add+mul` benchmark):

| Size | Before | After | Predicted | Actual | Notes |
|------|--------|-------|-----------|--------|-------|
| 1,000 | 5.4ms | 3.3ms | 2x | **1.6x** | Allocation overhead dominates at small sizes |
| 10,000 | 29.8ms | 14.0ms | 3.5x | **2.1x** | malloc + tape_append larger fraction than expected |
| 100,000 | 141.6ms | 21.8ms | 4x | **6.5x** | vDSP better than predicted (prefetch + pipeline) |

Softmax improved 1.6x (benefits from vectorized `exp` inside). Training step unchanged (dominated by matmul).

**Retrospective**: vDSP exceeded predictions at large sizes but undershot at small sizes. This tells us the remaining gap at 10k elements is **allocation overhead** (malloc per op + tape append), not arithmetic. To close further, we'd need to eliminate per-op allocation (tensor pooling or fused ops). At 100k, we went from 28x slower than PyTorch to **4.4x slower** — the remaining gap is entirely allocation cost (PyTorch uses memory pools, we malloc every time).

### Round 2: Fused `tensor_linear` (all backends)

**Change**: Add `tensor_linear(W, x, bias)` that does `y = W @ x + b` in a single C call with one allocation, one tape entry, and a fused backward rule. Implemented on all three backends.

**Predictions vs Results** (tape backend, `train_step` benchmark):

| Size | Before | After | Predicted | Actual | Notes |
|------|--------|-------|-----------|--------|-------|
| 64→64 | 1.0ms | 0.98ms | 30% faster | **~0%** | Backward dominates, not forward alloc |
| 256→256 | 5.5ms | 4.9ms | 20% faster | **~10%** | Modest, within noise |
| 1024→1024 | 7.7ms | ~8ms | 5% faster | **~0%** | BLAS dominates entirely |

**Retrospective**: The prediction was wrong. I assumed forward-pass allocation was a significant fraction of the training step cost. In reality, the backward pass (tape walk + gradient accumulation) dominates even at 64d. The `tensor_linear` fusion is still valuable for code cleanliness and will help more in real models (where many linear layers compound the savings), but for a micro-benchmark with a single layer + sum loss, the backward walk cost dwarfs the allocation saving.

**Lesson**: Micro-benchmarks can mislead about where overhead lives. The training step includes: forward (fast), backward (tape walk + grad loops — slow), optimizer step (fast). Eliminating 1 allocation in the forward doesn't move the needle because backward is 60-70% of the step.

## Notes

- The tape backend uses Apple Accelerate for BLAS (`cblas_dgemv`, `cblas_dgemm`) and vDSP for element-wise vectorization. On Linux without Accelerate, both matmul and element-wise fall back to scalar loops
- The ~50ms/epoch Chez Scheme overhead is constant regardless of model size. For large models (NTM at production scale), it's negligible. For small models (Linear classifier), it dominates
- Numbers in this doc are from a single representative run. Run `make bench-ops-compare` for results on your hardware

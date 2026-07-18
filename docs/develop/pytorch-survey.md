# PyTorch internals survey — Job 1 Phase B / Job 2b Phase B

> **Historical record.** Identifiers and paths reflect the tree at the time of
> writing; not updated for later renames (Executor spellings 2026-06-06, `Ml.*`
> module nesting 2026-07-27). Name decoder: [path-c-migration.md](path-c-migration.md).

Survey of `pytorch/pytorch` (source + design docs + release notes) for
portable performance patterns. Phase B deliverable per the plan.

Date: 2026-05-12. CPU-only target; primary platform macOS+Accelerate;
single-threaded eager (no JIT, no torch.compile).

## TL;DR

| Rank | Pattern | What | Effort | Impact |
|---|---|---|---|---|
| 1 | Small-shape matmul fast path | Bypass cblas setup below `M*N*K ≤ ~512`; NEON 4×4 register tile | ~250 LOC | high (NTM 4×4 heads) — **PyTorch itself doesn't ship this** |
| 2 | `tensor_softmax_xent` fused fwd+bwd | Backward is `(softmax − one_hot)/N` — no separate softmax-backward pass | ~150 LOC | medium (classification heads) |
| 3 | Per-size free-list for `Tensor*` metadata | Caching allocator on the fixed-size metadata struct only; data buffers already arena | ~80 LOC | medium (hot tape loops) |
| 4 | `InferenceMode` upgrade in `backend_torch.cpp` | Strictness + small per-op savings | trivial | low |
| 5 | Vec helper for ops not on the vDSP fast path | We're already on vDSP/`vvexp`/`vvtanh` for same-shape; gaps elsewhere | ~200 LOC | low-medium |
| 6 | `AccumulateGrad` "steal first grad" | Avoid clone on first backward step per param | ~30 LOC | very low |

Items 1-3 are the load-bearing Phase B work for Job 2b. Item 4 is the
only Job 1 win the survey surfaces.

## The "absolute speed" question

The 2026-05-09 sweep ratios on DNC-family (9.8×, 14.2×) aren't a
fixable libtorch-binding bug — they're per-op-count math.

One DNC timestep in PyTorch ref is ~30-40 `nn.*` calls (each a single
fused libtorch kernel; one wrapper trip, one autograd-graph node).
The same timestep in idris-ml expands into ~80-100 `prim__*` calls
because our `Layer.*` modules compose finer-grained primitives. With
T=20, that's 1600+ prim trips × the ~9 µs/prim Idris-side glue floor
from `ProfileMicro.idr` = ~14 ms/epoch of pure Idris dispatch — on
the same order as PyTorch ref's whole epoch.

PyTorch's "absolute speed" on DNC-class isn't kernel quality. It's
op granularity. To close it we have to reduce our per-epoch dispatch
count, not make individual primitives faster.

**Three levers, increasing effort:**

1. **Layer-level fused tape ops.** Generic (not arch-specific):
   `softmax_xent`, fused `attention`, more `OP_LSTM_GATES`-style nodes.
   Already done extensively (`OP_LAYER_NORM_2D`, `OP_LSTM_GATES`,
   `OP_LSTM_GATES_CELL`, `tensor_lstm_cell`, `tensor_gru_cell`,
   `tensor_softmax_{2d,3d}`). Plan's "out of scope: architecture-
   specific fused C ops" was about preventing `ntm_cosine_softmax_mul`,
   not about preventing generic library-level fusions.
2. **Reduce per-prim wrapper cost.** The 9 µs floor is part Idris
   codegen (out of scope, separate compiler project) and part our
   wrapper. Wrapper share is bounded; some room remains.
3. **Bigger workloads.** Dispatch tax amortizes when compute per prim
   grows. Closes ratio without fixing cause. (Job 2b direction-of-
   travel note.)

## What we already do

Important: the agent that did the GitHub walk wasn't aware of our
current state. Verifying claims against `backend_tape.c` reveals
substantial existing infrastructure:

**Fused tape ops we already have:**
- `OP_LSTM_GATES` + `OP_LSTM_GATES_CELL` (gate FCs + sigmoid/tanh +
  cell update in one tape node).
- `OP_LAYER_NORM_2D` (RowwiseMoments-style single-pass forward).
- `OP_SOFTMAX_2D` / `OP_SOFTMAX_3D` (max-reduce + exp + normalize
  fused; not unrolled into separate ops).
- `tensor_lstm_cell` / `tensor_gru_cell` / `tensor_cosine_similarity`
  (composite layer helpers).

**Vectorized elementwise (already there, not "we run scalar"):**
- Same-shape `binop_elementwise` uses `vDSP_vaddD` / `vDSP_vsubD` /
  `vDSP_vmulD` / `vDSP_vdivD` (`backend_tape.c:724-729`).
- Negate uses `vDSP_vnegD`; transcendentals use `vvexp` and
  `vvtanh` (`backend_tape.c:823-827`).
- Linear bias-add uses `vDSP_vaddD` (`backend_tape.c:1054, 1123`).

**Memory:**
- Chunked arena allocator for data buffers (per project memory:
  "Arena realloc invalidates pointers" — using linked list of chunks).
- `Tensor` metadata structs are NOT pooled — each goes via `malloc`/
  `free`. This is the opportunity Item 3 below targets.

## Surveyed patterns

### Pattern 1 — Small-shape matmul fast path (Job 2b, top priority)

**Surveyed**: `aten/src/ATen/native/CPUBlas.cpp`,
`aten/src/ATen/native/cpu/BlasKernel.cpp`, `LinearAlgebra.cpp`.

**What PyTorch does**: dispatches every CPU matmul to `cblas_sgemm`
(or its templated triple-loop fallback when transposes don't fit the
BLAS signature). **No shape-conditional fast path** for small
matmuls. Quote from the agent: "PyTorch's strategy is *trust BLAS,
fall through to a generic triple-loop otherwise*. This is consistent
with their benchmarks — they don't run on tiny shapes."

**Why this matters for us**: idris-ml routinely hits 4×4 matmuls
(NTM heads, small attention). BLAS setup cost per `cblas_dgemm` call
is meaningful at those sizes; PyTorch has the same overhead but
nobody notices because their CPU benchmarks don't hit it.

**Port**: in `tensor_mm` / `tensor_bmm`, branch on `M*N*K`:
- `≤ ~32`: fully unrolled triple-loop. Compile-time-known shapes
  could collapse further if we route through type-driven specializers.
- `≤ ~512`: hand-rolled triple-loop with NEON FMA (`vfmaq_f32`,
  `vfmaq_laneq_f64` on M1+ for double), 4×4 register tile.
- Else: `cblas_dgemm`.

Threshold tuning via the existing `bench_ops.c` harness. Effort:
~250 LOC. Impact: large on NTM (4×4 heads), and indirectly on
DNC's content addressing (small cosine-sim matmuls).

**Sources**:
- `aten/src/ATen/native/CPUBlas.cpp`
- `aten/src/ATen/native/cpu/BlasKernel.cpp`

### Pattern 2 — `tensor_softmax_xent` fused fwd+bwd (Job 2b)

**Surveyed**: `aten/src/ATen/native/cpu/SoftMaxKernel.cpp`,
`aten/src/ATen/native/LossNLL.cpp`.

**What PyTorch does**: `F.cross_entropy(logits, target)` is a single
op that fuses log_softmax + NLL forward and produces gradients via
the closed-form `grad = (softmax − one_hot) / N` — no separate
softmax-backward pass through the tape.

**What we have**: `tensor_cross_entropy` (`backend_tape.c:1615`)
exists but is **no-grad** — comment reads "simplified, no grad". So
training paths through classification heads have to decompose into
`tensor_log_softmax → ... → tensor_sum`, which is multiple tape
nodes with separate backward passes.

**Port**: add `tensor_softmax_cross_entropy_with_logits(logits,
target)` as a single fused op with backward rule `(softmax − target)
/ N`. Numerically stable (single max-subtract pass). One tape node
replaces 3-4. Effort: ~150 LOC. Impact: classification heads —
MNIST, any future supervised example.

Note: this is structurally similar to the `OP_LSTM_GATES` pattern
we already use — single tape op with internal multi-step forward
and closed-form backward.

**Sources**:
- `aten/src/ATen/native/cpu/SoftMaxKernel.cpp`
- `aten/src/ATen/native/LossNLL.cpp`

### Pattern 3 — Per-size free-list for `Tensor*` metadata (Job 2b)

**Surveyed**: `c10/mobile/CPUCachingAllocator.cpp`,
`c10/core/CPUAllocator.cpp`.

**What PyTorch does**: the **default** CPU allocator is just
`posix_memalign` / `free` — no caching. There is an opt-in
`CPUCachingAllocator` in mobile-only paths; implementation is
exact-size bucket via `ska::flat_hash_map<size_t, SmallVector<void*,
16>>` keyed on bytes, with a global `std::mutex`. Header literally
says: *"experimental and might disappear in the future."*

**Signal**: even PyTorch's own implementation is exact-size, global-
mutex, mobile-only. They don't bother with size buckets or
thread-local pools because their CPU workload is alloc-light. For
us — every tape node allocates a `Tensor*` — this matters more.

**Port**: per-size free-list specifically for the fixed-size
`Tensor` metadata struct. Data buffers already arena-allocated.
Free path on `tape_reset`: push struct onto free-list. Allocate
path: pop from free-list or `malloc` if empty.

The fixed-size case is much simpler than PyTorch's variable-size
hash-keyed one — no map, just a single linked free-list. ~80 LOC
including the integration with the tape reset cycle.

Effort: ~80 LOC. Impact: removes `malloc`/`free` from the per-prim
hot path on NTM-copy (50 tape nodes/timestep × T=20 = 1000+ alloc/
free pairs per epoch).

**Sources**:
- `c10/mobile/CPUCachingAllocator.cpp`
- `c10/core/CPUAllocator.cpp`

### Pattern 4 — `InferenceMode` in `backend_torch.cpp` (Job 1)

**Surveyed**: `c10/core/InferenceMode.h`, `autograd_meta.cpp`.

**What**: replace `torch::NoGradGuard` with `c10::InferenceMode` in
`optimizer_step` (and any other forward-only-no-backward paths in
`backend_torch.cpp`).

**Why**: `InferenceMode` is strictly stronger than `NoGradGuard` —
skips version-counter bumps and view-tracking *in addition* to
disabling autograd. Marginal per-op savings; cumulative across
optimizer-step's many tensor operations.

**Risk**: low but non-zero. `InferenceMode`-produced tensors can't
later enter autograd unless explicitly cloned. Optimizer outputs
are in-place param updates — they never enter autograd — so safe.

Effort: ~3 lines. Impact: too small to show in `perf-baseline.md`
but strictly an improvement.

**Sources**: `c10/core/InferenceMode.h`

### Pattern 5 — More vec coverage (Job 2b, conditional)

**Surveyed**: `aten/src/ATen/cpu/vec/`, especially
`vec128/vec128_float_neon.h`.

**What PyTorch does**: hand-rolled NEON intrinsics with SLEEF /
ARM-Optimized-Routines polynomial approximations for `exp`/`log`/
`sin`/`cos`/`tanh`. Template `Vectorized<float>` provides a uniform
interface across NEON/AVX2/AVX512/AVX-512-FP16.

**What we have**: vDSP / `vvexp` / `vvtanh` from Accelerate on
same-shape elementwise. The agent's claim that "we run scalar" is
wrong for the same-shape fast path.

**Gap**: ops that *don't* hit the same-shape vDSP fast path. The
`binop_elementwise` scalar/broadcast paths probably run a scalar
loop. Reductions (sum, mean, var) are scalar. Sigmoid is `1/(1+exp(-x))`
— could fuse via a NEON `_vec_sigmoid4` to save a pass.

**Decision**: defer until measurement justifies it. Benchmark
Accelerate's `vvexp`/`vvtanh` vs hand-rolled NEON polynomial at our
typical tensor sizes (often <1K elements) before assuming hand-
rolled wins. Accelerate's call overhead might matter more than its
inner-loop quality at these sizes — measure first.

**Sources**: `aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h`

### Pattern 6 — `AccumulateGrad` steal-first-grad (Job 2b)

**Surveyed**: `torch/csrc/autograd/functions/accumulate_grad.h:228-257`.

**What**: when a leaf grad doesn't yet exist, PyTorch *steals* the
incoming gradient tensor rather than cloning, provided no other ref
points at it and double-backward isn't requested.

**For us**: tape's gradient accumulation probably already initializes
grads to zero-tensors, then does in-place `+=`. The PyTorch "steal
first" trick is specifically about avoiding the first clone when grad
*doesn't yet exist*. Our pattern likely doesn't have that path —
need to verify in `backend_tape.c`'s optimizer step.

If verified absent, ~30 LOC; saves one clone per parameter per
backward.

**Sources**: `torch/csrc/autograd/functions/accumulate_grad.h`

## Skipped (and why)

- **Dispatcher / `OperatorEntry` kernel cache** — already O(1) flat
  array lookup; no wrapper-side win available (verified in headers).
- **`cblas_dgemm_batch_strided`** — verified missing from macOS
  Accelerate (`MacOSX15.5.sdk/.../cblas_new.h`). Only available on
  Linux/OpenBLAS/MKL paths; would only help CI runs.
- **Fused LSTM/GRU cell** — we already have `OP_LSTM_GATES` and
  `tensor_lstm_cell`/`tensor_gru_cell`. PyTorch itself only fuses on
  CUDA; on CPU they decompose into separate ops (verified via direct
  fetch of `aten/src/ATen/native/RNN.cpp`). **We're already winning
  here vs libtorch on CPU.**
- **Fused LayerNorm** — already have `OP_LAYER_NORM_2D`.
- **TensorImpl pool allocator** — structural, too large for Phase B.
- **`TensorIterator` framework** — code-quality refactor; the perf
  win lives in Pattern 5 (vec helpers), not in the iterator
  abstraction itself. Defer.
- **Compiled Autograd / `torch.compile` / Inductor** — requires a
  tracing layer we don't have. Out of scope.
- **Channels-last memory format** — helps batched conv at ImageNet
  scale; our conv path is single-batch toy MNIST. Skip.
- **Caching allocator beyond `Tensor*` metadata** — data buffers
  already arena-allocated; nothing to cache.

## Recommended porting order

1. **Pattern 1 (small-shape matmul fast path)** — direct attack on
   NTM/DNC 4×4-head dispatch cost. PyTorch doesn't ship this, so the
   win is genuinely above libtorch's CPU floor on those workloads.
   Closing convergence sweep on tape + torch + mlx after landing.
2. **Pattern 3 (per-size `Tensor*` free-list)** — small isolated
   patch, hot-path malloc reduction. Land in parallel with 1; doesn't
   interact.
3. **Pattern 2 (`tensor_softmax_xent` fused)** — closes one of the
   remaining classification-head decompositions. Lower priority since
   MNIST is the only example exercising it heavily; revisit when a
   larger classification workload lands.
4. **Pattern 4 (`InferenceMode` swap)** — drive-by improvement in
   `backend_torch.cpp`. Trivial.
5. **Pattern 6 (steal-first-grad)** — verify it's actually absent;
   if so, micro-win.
6. **Pattern 5 (more vec coverage)** — measure first; defer unless
   profiling justifies.

Each pattern lands as its own commit so the closing-sweep gate can
attribute movement.

## Future fused ops surfaced by the precedent test

Per the `pytorch_precedent_test` feedback memory, a fused C op is
in scope iff PyTorch ships an equivalent. Patterns the survey
surfaced that pass the test but don't justify Phase B effort
*today* (no example exercises them heavily) — file as Phase C
candidates if/when a relevant workload lands:

- **`tensor_scaled_dot_product_attention`** — PyTorch:
  `F.scaled_dot_product_attention` + the CPU FlashAttention
  kernel referenced above. ~600 LOC; revisit when a transformer
  example justifies it.
- **`tensor_rms_norm`** — PyTorch: `torch.nn.RMSNorm`. Standard
  in modern LLM stacks. ~100 LOC; revisit if an LLM-style example
  lands.
- **`tensor_silu`, `tensor_gelu_linear`, `tensor_silu_linear`** —
  PyTorch: `F.silu`, fused gate+linear patterns from `torch._foreach`
  and inductor. Small; bundle with vec helper work if measurement
  surfaces them.
- **`tensor_group_norm`** — PyTorch: `nn.GroupNorm`. Useful for
  small-batch image models. Revisit when one lands.

The point of listing these is to record that they're *available*
to reach for, not blocked behind a "no architecture-specific ops"
rule. Order of operations stays measurement-driven.

## Sources

GitHub references (libtorch 2.9 / main):
- [TensorIterator](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.h)
- [cpu_kernel_vec / Loops.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/Loops.h)
- [Vec backend (NEON)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h)
- [LayerNorm kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)
- [SoftMax kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)
- [FlashAttention CPU](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp)
- [RNN.cpp (LSTM/GRUCell — CPU unfused)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/RNN.cpp)
- [CPUBlas / BlasKernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/CPUBlas.cpp)
- [CPUCachingAllocator (mobile-only)](https://github.com/pytorch/pytorch/blob/main/c10/mobile/CPUCachingAllocator.cpp)
- [AccumulateGrad](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/functions/accumulate_grad.h)
- [intrusive_ptr](https://github.com/pytorch/pytorch/blob/main/c10/util/intrusive_ptr.h)
- [Inductor CPP codegen](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/cpp.py)
- [PyTorch 2.5 release notes](https://pytorch.org/blog/pytorch2-5/)

Local libtorch headers walked:
- `Dispatcher.h`, `OperatorEntry.h`, `CPUBlas.h`, `accumulate_grad.h`,
  `engine.h`, `function.h`, `input_buffer.h`, `TensorImpl.h`,
  `InferenceMode.h`.

# C coverage: remaining gaps + plan

Snapshot after the 2026-06-26 per-file gap audit (gcov + gcovr, product-only, all
three lanes green). This is the planning companion to
[`coverage-policy.md`](coverage-policy.md) — it enumerates what is still
uncovered, why, and how to close it. Re-measure with `make test-coverage-all`
(per-backend overview) or `make BACKEND=<b> test-coverage-backend` then read
`build-cov-<b>/html/`.

## Current state

| backend | lines | branches | suite |
|---------|-------|----------|-------|
| tape    | 98.0% | —        | 549     |
| mlx     | 99.2% | —        | 632     |
| torch   | 95.0% | —        | 544     |

(2026-06-26 progress: Task-1 exclusions lifted mlx 94.1% → 98.3% — NAN_TRAP +
MLX_OPT_COMPILE marked `GCOVR_EXCL`, no behaviour change. Task-2 optimizer tests
lifted torch 90.9% → 94.7% — new `test_optimizer_torch.c` + common shared-trampoline
tests in `test_optimizer_tape.c`. The optimizer files on all three backends —
`optimizer.cpp` (torch/mlx), `optimizer.c` (tape) and `shared/training/optimizer.c`
— are now fully covered or `GCOVR_EXCL`.)

(Task-3 Bucket-C wave: conv / softmax / gru / linalg / cat2 (tape),
dtype_dispatch / create_param_state / autograd (mlx), bitlinear / adapter /
dtype_dispatch (torch) backfilled — tape 96.7% → 98.0%, mlx 98.3% → 99.2%,
torch 94.7% → 95.0%. The wave surfaced **two real bugs** (Bucket A): a tape
`conv2d_batched` F32 backward heap overflow — `r->grad`/`a->grad`/`b->grad`
read/written as `double*` on float-sized F32 grad buffers (fixed: widen via the
typed `tape_grad_*_d` accessors) — and an mlx `tensor_item_1d` non-contiguous
read — `mx_read_double` indexed raw storage, ignoring strides, so a multichannel
conv1d output came back transposed (fixed: flatten before reading, matching
`tensor_item_2d`). Both were latent: multichannel/F32 conv backward had no prior
test on any backend.)

(Earlier `TAPE_ABORT_IF` rollout: the standalone `if(cond){fprintf;abort;}` guards
in `op_dispatch.c` / `cat2.c` / `unsqueeze.c` (plus the earlier `bitlinear.c`
conversion) now use the same-line-guard macro — covered with no `GCOVR_EXCL` and no
coverage-only death test. else-branch / switch-default aborts stay `GCOVR_EXCL`.)

(Up from baselines tape 79.2% / mlx 78.8% / torch 73.7%.) Branch% is intentionally
lower — gcov counts many compiler-generated/defensive branches; line% is the
chased metric.

**Measurement note (2026-06-26):** tape previously *mis-read* as 92.8% (6402/6899)
because a prior multi-link coverage build (`BACKEND=tape,mlx,torch`) left
`shared_training_{mlx,torch}/` `.gcno` (compiled, never run) under
`build-cov-tape/`; gcovr triple-counted each shared TU, inflating the denominator
(`param_registry.c`/`ffi_shims.c` showed ~50% when fully covered). The
`test-coverage-backend` self-heal now purges every non-primary `*_<b>/` tree, so
this can't recur. **torch genuinely regressed** to 90.9% — the fused
`adamw_step_foreach` / `rmsprop_step_foreach` paths in `optimizer.cpp` landed
without coverage tests (the suite drives only Adam). See Bucket C.

## Per-file gap audit (2026-06-26)

Top product gaps by backend (parse `build-cov-<b>/cov.xml`; full list via the
audit one-liner in git history). The single largest *real* gap is the optimizer
variants — common to all three backends.

| Lines | File | Backend | Category |
|---|---|---|---|
| ~~109~~ → 38 | `backend_torch/training/optimizer.cpp` | torch | **C (done) — new `test_optimizer_torch.c`: adam/adamw/rmsprop foreach, clip-norm rescale, set_param_lr, m/v + meta round-trip, step_with_clip, scaled** |
| ~~103~~ → 4 | `backend_mlx/training/optimizer.cpp` | mlx | **B (done) — MLX_OPT_COMPILE compile path now `GCOVR_EXCL`; eager paths already covered** |
| ~~69~~ → 3 | `backend_mlx/training/backward.cpp` | mlx | **B (done) — `DEBUG_NAN_TRAP=1` diagnostic now `GCOVR_EXCL`** |
| 42 | `backend_tape/nn/quantization/bitlinear.c` | tape | mixed — F32 absmean/quant arms (C) + `abort()` guards (B) |
| 32 | `backend_tape/tape.c` | tape | C(scale) — arena multi-chunk growth (>64K tape entries) + reset/meta |
| 21 | `backend_torch/nn/quantization/bitlinear.cpp` | torch | C — BitNet quant edge arms |
| 20 | `backend_mlx/training/dtype_dispatch.cpp` | mlx | mixed — float dtag arms (C) + integer-param aborts (B) |
| 15 | `backend_mlx/core/lifecycle/create_param_state.cpp` | mlx | C — param/state creator edge arms |
| 15 | `backend_torch/training/adapter.cpp` | torch | C — adapter edge paths |
| 12 | `shared/training/optimizer.c` | tape | C — shared optimizer helper arms |
| 11 ea | `conv2d_batched.c` / mlx `autograd.cpp` / torch `dtype_dispatch.cpp` | — | C — conv backward + dtag arms |
| ≤10 | conv1d/conv2d, linear_2d, mv, cat2, softmax, gru_cell, pools, scalar ops, … | tape | C — last-mile backward/edge branches (long tail) |

The remaining ~6-7% per backend falls into the three buckets below.

## Bucket A — coverage-surfaced bugs (3 fixed, 1 mlx-upstream)

All four were investigated 2026-06-25 (ASan-pinpointed). See CHANGELOG.

| Area | Status |
|---|---|
| general-broadcast elementwise (`[2,3]+[3]`) | **FIXED** — was a *test* bug (stack array freed by `tensor_create_2d`'s callee-owns contract), not a product defect. The general-broadcast path is correct; the 6 tests re-enabled. |
| BitNet quant `absmean`/`ternary_quant` (all 3) | **FIXED** — test bugs (`create_2d(stack)`, 1-byte `hf` for `[1,4]`) **plus a real tape product bug**: `round_clamp_ternary` was round-half-AWAY, not round-half-to-even; fixed to `nearbyint` (matches torch/mlx/PyTorch). 8 tests re-enabled. |
| `tensor_pair_free` (all 3) | **FIXED** — dead double-free-prone API removed entirely; pairs are reset-owned. |
| `optimizer.cpp` MLX_OPT_COMPILE Adam branch (mlx) | **WON'T-FIX (mlx upstream)** — `mx::compile` runs (g++ shim) but libmlx's Metal-allocator/device static teardown crashes the forked child; clearing our cache + `mx::detail::compile_clear_cache()` both insufficient (cf. `init.cpp`). The 2 `mlx_optimizer_compile` tests stay `.disabled`; this branch is a **principled exclusion** below until mlx fixes the teardown. |
| tape `conv2d_batched` F32 backward heap overflow | **FIXED (2026-06-26)** — `tape_backward_conv2d_batched` read `r->grad` and wrote `a->grad`/`b->grad` directly as `double*`, but F32 tensors get float-sized grad buffers (`ensure_grad`). On an F32 conv the result-grad read overran 4 bytes/elem and the Apple cblas d_kernel / col2im d_input writes overran. Fix: widen through the typed `tape_grad_*_d` accessors. ASan-pinpointed by the new multichannel-F32 conv test (no prior F32 conv backward test on any backend). |
| mlx `tensor_item_1d` non-contiguous read | **FIXED (2026-06-26)** — read `mx_read_double(t->data, idx)`, which indexes raw storage ignoring strides; an mlx transpose is a lazy strided view, so a multichannel conv1d output (`[outC,oL]` via internal transpose) read back as `[oL,outC]`. Fix: flatten to contiguous before reading, matching `tensor_item_2d`. Latent because single-element-per-axis results are layout-invariant. Surfaced by the new multichannel conv1d test. |

## Bucket B — principled exclusions (no CI input reaches them)

Already marked with inline `GCOVR_EXCL_*` (reason at the line) or excluded at file
level in `gcovr.cfg` / `codecov.yml`. Not counted against the number.

- **GPU/MPS/CUDA-only paths** on the CPU CI lanes: `mlx/init.cpp` (GPU device
  setup + terminate handler), `torch/mps_init.cpp`, `torch/device.cpp` &
  `backend_meta.cpp` non-CPU branches. The mlx lane is mlx-cpu; the torch lane is
  torch-cpu.
- **Own-line `abort()` guards**: the `abort()` itself can't be line-covered (it
  skips the gcov flush even under a `.signal=SIGABRT` death test), so the fatal
  body carries `GCOVR_EXCL_*` with the death test named in the reason. The
  *guard condition* is covered. (~10-15 sites across the three.)
- **F32-only kernels on tape** are NOT excluded — they're covered via the streamed
  dtag-14 path (tape's bare `tensor_create_*_f32` aborts; F32 lives on streamed).
  Exception: `bitlinear.c`'s `tensor_absmean_per_row_2d` / quant F32 arms are NOT
  on the streamed path and are currently uncovered (~20 lines) — testable with a
  direct F32 input (Bucket C), not an exclusion.
- **`DEBUG_NAN_TRAP=1` diagnostic** (`backend_mlx/training/backward.cpp` body of
  the `if(env)` block): env-gated NaN-locating tape walk, runs only when the env var
  is set *and* a NaN is present. **DONE 2026-06-26** — wrapped `GCOVR_EXCL`
  (backward.cpp 69→3 miss). The `getenv` condition itself stays counted.
- **`MLX_OPT_COMPILE` Adam-compile path** (`backend_mlx/training/optimizer.cpp`
  `get_adam_compiled` / `adam_step_compile` / the compile branch in
  `optimizer_step`): won't-fix on CI (libmlx Metal teardown crashes the forked
  test child); pinned by the `.disabled` `mlx_optimizer_compile` tests. **DONE
  2026-06-26** — wrapped `GCOVR_EXCL` (optimizer.cpp 103→4 miss). The eager
  Adam/AdamW/RMSprop/SGD paths stay counted and covered.
- **Diagnostics / dispatch-table init / vendored / .venv framework headers**:
  file-level via `gcovr.cfg`.

## Bucket C — testable but deferred (reachable; just not yet written)

Reachable via the public FFI on the CI lane; the coverage workflows didn't
exhaust them. Ordered by leverage:

| Lines | Area | Note |
|---|---|---|
| **optimizer.cpp (torch -109, mlx -103) + shared `optimizer.c` (tape -12)** | **optimizer variants — the dominant real gap** | The coverage suites drive only **Adam**. Uncovered: `optimizer_create_adamw` ctor, `adamw_step_foreach` (decoupled wd), `rmsprop_step_foreach`, SGD/momentum, clip-by-value, per-group LR, schedule `tick`, `optimizer_set_m/v` realloc, load/save state. **One focused optimizer test file per backend closes ~220 lines** and recovers torch's regression. Highest leverage by far. |
| tape `bitlinear.c` F32 arms -~20 / torch `bitlinear.cpp` -21 | BitNet quant edge | `absmean_per_row_2d` / quant F32 arms (direct F32 input) + edge branches. The `abort()` guards in the same files are Bucket B. |
| tape `conv2d_batched` -11 / `conv2d` -10 / `conv1d` -7 | conv backward + edge | stride/padding/dilation variants + backward grad asserts. |
| `dtype_dispatch` (mlx -20, torch -11) | remaining dtag arms | cover float dtags; I32/Bool param-leaf arms abort on torch (`make_param_leaf` always sets requires_grad → c10 throw) — leave integer-param as a guard/exclusion. |
| mlx `create_param_state.cpp` -15 / `autograd.cpp` -11, torch `adapter.cpp` -15 | creator + adapter edge | param/state creator dtag arms + adapter paths. |
| tape `tape.c` -32 | tape mechanics (scale) | arena multi-chunk growth fires only past 64K tape entries — testable only with a large workload; low-leverage. Reset/meta paths are cheap. |
| small: `mv`/`linear`/`linear_2d`/`cat2`/`softmax`/`log_softmax`/`gru_cell`/pools/scalar (≤9 each) | misc backward/edge | last-mile branches; volume. |

Closing Bucket C would push each backend into the high 90s. None is hard; it's
volume. A follow-up workflow wave (same `.tmp/coverage-workflow-*.js` templates)
targeting optimizer + bitlinear F32 + conv would capture most.

**Status (2026-06-26 Task-3 close).** The Bucket-C wave landed all of the above
**except** tape `tape.c` arena multi-chunk growth (only reachable past 64K tape
entries — testable only at scale, low ROI; **deferred**, a known hole). Final:
tape 98.0% / mlx 99.2% / torch 95.0%. torch's smaller bump reflects that its
dtype arms are runtime-dtyped on shared source lines (F32/F64 traverse the same
lines), so the new tests add branch/behaviour coverage more than line coverage.

## Recommended order

1. ✅ **Done** — Bucket A bugs investigated + fixed (general-broadcast, BitNet
   quant rounding, `tensor_pair_free`); mlx compile-teardown is won't-fix
   (upstream). See CHANGELOG.
2. **Mark the `DEBUG_NAN_TRAP` block + remaining `abort()` guards `GCOVR_EXCL`**
   (Bucket B) — cheapest honest lift (~62 mlx lines), no new tests.
3. **One optimizer coverage test file per backend** (RMSprop/SGD/AdamW/clip/wd/
   per-group/save-load) — the single biggest real gap (~220 lines across the
   three), and it recovers torch's regression to ≥93%.
4. One Bucket-C workflow wave (bitlinear F32, conv, dtype_dispatch, long tail) for
   the high-90s — the remaining gap is volume, not hard.
5. (Blocked on mlx upstream) the MLX_OPT_COMPILE branch — re-enable the 2
   `mlx_optimizer_compile` tests once mlx's compile/Metal teardown is stable.

## How the gate behaves meanwhile

Codecov (`codecov.yml`) gates **patch coverage at 100%** (new C must be covered or
inline-excluded) and **project coverage with a 1% threshold** per flag. So the
current numbers are the floor — they can't regress, and Bucket A/C work ratchets
them up. The four-axis gap probe (`make coverage-gap-probe`) remains the
complementary symbol/OP_*/F32-oracle gate.

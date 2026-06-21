# C coverage: remaining gaps + plan

Snapshot after the 2026-06-25 coverage push (gcov + gcovr, product-only, all three
lanes green). This is the planning companion to
[`coverage-policy.md`](coverage-policy.md) — it enumerates what is still
uncovered, why, and how to close it. Re-measure with
`make BACKEND=<b> coverage-backend-<b>` and read `build-cov-<b>/html/`.

## Current state

| backend | lines | branches | suite |
|---------|-------|----------|-------|
| tape    | 94.3% | 74.5%    | 501/501 |
| mlx     | 93.3% | 49.9%    | 555/555 |
| torch   | 93.2% | 50.0%    | 470/470 |

(Up from baselines tape 79.2% / mlx 78.8% / torch 73.7%.) The remaining ~6-7% per
backend falls into three buckets below. Branch% is intentionally lower — gcov
counts many compiler-generated/defensive branches; line% is the chased metric.

## Bucket A — coverage-surfaced bugs (3 fixed, 1 mlx-upstream)

All four were investigated 2026-06-25 (ASan-pinpointed). See CHANGELOG.

| Area | Status |
|---|---|
| general-broadcast elementwise (`[2,3]+[3]`) | **FIXED** — was a *test* bug (stack array freed by `tensor_create_2d`'s callee-owns contract), not a product defect. The general-broadcast path is correct; the 6 tests re-enabled. |
| BitNet quant `absmean`/`ternary_quant` (all 3) | **FIXED** — test bugs (`create_2d(stack)`, 1-byte `hf` for `[1,4]`) **plus a real tape product bug**: `round_clamp_ternary` was round-half-AWAY, not round-half-to-even; fixed to `nearbyint` (matches torch/mlx/PyTorch). 8 tests re-enabled. |
| `tensor_pair_free` (all 3) | **FIXED** — dead double-free-prone API removed entirely; pairs are reset-owned. |
| `optimizer.cpp` MLX_OPT_COMPILE Adam branch (mlx) | **WON'T-FIX (mlx upstream)** — `mx::compile` runs (g++ shim) but libmlx's Metal-allocator/device static teardown crashes the forked child; clearing our cache + `mx::detail::compile_clear_cache()` both insufficient (cf. `init.cpp`). The 2 `mlx_optimizer_compile` tests stay `.disabled`; this branch is a **principled exclusion** below until mlx fixes the teardown. |

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
- **Diagnostics / dispatch-table init / vendored / .venv framework headers**:
  file-level via `gcovr.cfg`.

## Bucket C — testable but deferred (reachable; just not yet written)

Reachable via the public FFI on the CI lane; the coverage workflows didn't
exhaust them. Ordered by leverage:

| Lines | Area | Note |
|---|---|---|
| mlx `backward.cpp` -69 | mlx replay vjp arms | each `MLX_REGISTER_REPLAY` op needs a forward+backward to exercise its replay; the common tests hit the common ops, not all. Drive the long tail (conv/pool/norm/attention replays). |
| optimizer.cpp tail (mlx -103 minus compile, torch -63, tape -12) | optimizer variants | clip-by-value, weight-decay edges, per-group LR, schedule `tick`, `optimizer_set_m/v` realloc arms, load/save state. |
| tape `conv2d_batched` -11 / `conv2d` -10 / `conv1d` -7 | conv backward + edge | stride/padding/dilation variants + backward grad asserts. |
| tape `tape.c` -32 | tape mechanics | tape growth/reset/meta paths not hit by small tests. |
| `dtype_dispatch` (mlx -20, torch -11) | remaining dtag arms | I32/Bool param-leaf arms abort on torch (`make_param_leaf` always sets requires_grad → c10 throw); cover float dtags, leave integer-param as a guard/exclusion. |
| tape `_kernels.inc` -50 (non-broadcast part) | binop default arm | the same-shape vDSP `default:` switch arm is reached by pow/min/max (not add/sub/mul/div) — add a pow/min/max same-shape test. |
| small: `mv`/`linear`/`linear_2d`/`cat2`/`softmax`/`log_softmax`/`gru_cell`/`accessors` (3-9 each) | misc backward/edge | last-mile branches. |

Closing Bucket C would push each backend into the high 90s. None is hard; it's
volume. A follow-up workflow wave (same `.tmp/coverage-workflow-*.js` templates)
targeting `backward.cpp` + optimizer + conv would capture most.

## Recommended order

1. ✅ **Done** — Bucket A bugs investigated + fixed (general-broadcast, BitNet
   quant rounding, `tensor_pair_free`); mlx compile-teardown is won't-fix
   (upstream). See CHANGELOG.
2. One Bucket-C workflow wave (mlx `backward.cpp`, optimizer tails, tape conv,
   `tape.c`) for the high-90s — the remaining gap is volume, not hard.
3. (Blocked on mlx upstream) the MLX_OPT_COMPILE branch — re-enable the 2
   `mlx_optimizer_compile` tests once mlx's compile/Metal teardown is stable.

## How the gate behaves meanwhile

Codecov (`codecov.yml`) gates **patch coverage at 100%** (new C must be covered or
inline-excluded) and **project coverage with a 1% threshold** per flag. So the
current numbers are the floor — they can't regress, and Bucket A/C work ratchets
them up. The four-axis gap probe (`make coverage-gap-probe`) remains the
complementary symbol/OP_*/F32-oracle gate.

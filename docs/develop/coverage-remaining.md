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

## Bucket A — bug-blocked (cannot cover until the bug is fixed)

These paths have committed `.disabled = true` reproducer tests; fixing the bug +
re-enabling the test closes the lines. All filed in [`TODO.md`](../../TODO.md).

| Area | Uncovered | Bug |
|---|---|---|
| `add.c`/`sub.c` general-broadcast backward (~35 lines) + `broadcast.c` general path (~16) + `_kernels.inc` general-broadcast block (~part of -50) | tape | **general-broadcast elementwise heap corruption** — `[2,3]+[3]` etc. crash. Reproducers: `core_elementwise_{add,sub}/*_general_broadcast_*`. Fixing unlocks ~50-65 tape lines. |
| `bitlinear.c` absmean/ternary_quant arms (part of tape -108, mlx -29, torch -21) | all 3 | **BitNet quant helpers crash** on valid 2D F64 (cross-backend). Reproducers: `nn_quantization_{absmean,ternary_quant}/*`. |
| `pair_helpers` free path (a few lines × 3) | all 3 | **`tensor_pair_free` crash** (cross-backend). Reproducers: `*_pair_helpers/*free*`. |
| `optimizer.cpp` MLX_OPT_COMPILE Adam branch (~part of mlx -103) | mlx | **mlx compile-path teardown crash** — the g++→clang++ shim makes `mx::compile` run but mlx's compile-state teardown crashes the forked child (no gcov flush). Reproducers: `mlx_optimizer_compile/*`. |

**Priority: the general-broadcast heap corruption is P1 (memory corruption).** The
others are crashes on narrower inputs. Fixing all four unlocks ~120-150 lines
across the three backends and removes the 18 disabled tests.

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

1. **Fix the P1 general-broadcast heap corruption** (Bucket A) — memory-safety,
   unlocks the most tape lines, removes 6 disabled tests.
2. Fix the BitNet quant + `tensor_pair_free` crashes (Bucket A) — cross-backend,
   removes 11 disabled tests.
3. One more Bucket-C workflow wave (mlx `backward.cpp`, optimizer tails, tape
   conv) for the high-90s.
4. The mlx compile-teardown crash (Bucket A) is lowest priority — opt-in feature,
   needs an mlx-internal teardown fix.

## How the gate behaves meanwhile

Codecov (`codecov.yml`) gates **patch coverage at 100%** (new C must be covered or
inline-excluded) and **project coverage with a 1% threshold** per flag. So the
current numbers are the floor — they can't regress, and Bucket A/C work ratchets
them up. The four-axis gap probe (`make coverage-gap-probe`) remains the
complementary symbol/OP_*/F32-oracle gate.

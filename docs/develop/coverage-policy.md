# Backend test coverage policy

This doc defines what "covered" means for the three backends
(`backend_tape`, `backend_torch`, `backend_mlx`), where coverage is
chased, where it's principally excluded, and the contributor
checklist when adding new ops.

## The four-axis target

A backend is fully covered when the **first three** axes hold; the
fourth is **additive** (confidence, not coverage *per se*):

1. **Symbol coverage** — every `extern "C"` entry in
   `packages/backends/backend.h` that is not on the exclusion list
   below has at least one Criterion test (now under
   `packages/backends/backend_{tape,torch,mlx}/<subsystem>/test_*.c`
   for colocated per-op tests + `packages/idris-test-c/src/` for
   cross-cutting infra) that exercises it.
2. **Backward coverage** — every `OP_*` tag in the backend's
   `tape.h` enum (`backend_tape/tape.h`, `backend_mlx/tape.h`) has a
   test that triggers its backward dispatch case and asserts gradient
   values against a hand-computed or finite-difference oracle. Torch
   has no `OP_*` enum (uses libtorch's autograd) and is covered via
   symbol coverage + W3b's custom-logic-path tests.
3. **F32 paired oracle** — every op routed through the F32 storage
   path has a paired F32-vs-F64 gradcheck rung in the tape T29 ladder
   (`packages/idris-test-c/src/test_legacy_backend.c`). This axis is
   tape-specific (mlx and torch's F32 paths live in their respective
   frameworks).
4. **Property-based confidence (additive, not gating).** When a
   kernel or pure-math operation has an invariant that fixed-shape
   tests can only sample-check (sum-to-one, norm-bounded, round-trip,
   F32-vs-F64 oracle), prefer a Hedgehog property in
   `packages/idris-ml/src/Test/Properties/*.idr` over a hand-coded
   fixed-shape test. Properties run via the `Test.Property` adapter
   from `idris-test` (`checkProperty "<name>" prop`); the underlying
   `Hedgehog.Property` is `PropertyT Identity`-based, so the
   *current* property surface is pure-math (softmax/rmsnorm
   formulas, shape arithmetic). FFI-driven properties — Hedgehog
   generators of shapes/values driving the C kernel through the
   FFI — need an IO-aware property runner; out of scope until a
   concrete need surfaces. Property tests do *not* replace Axes 1-3
   — they sit alongside, raising implementation confidence rather
   than coverage line count. Contributor checklist when adding a
   new OP_*: check whether the op has an invariant worth asserting
   beyond the fixed-shape oracle. If yes, file a `prop_*` in
   `Test/Properties/`. If not, the fixed-shape T29 oracle remains
   sufficient.

C-line coverage (`llvm-cov report` %) is a **secondary** metric —
recorded in HTML artifacts via `make coverage-backend-<b>`, but not
chased as a number. Chasing line % tempts tests that lift coverage
without lifting confidence (the dispatch-table init being the
canonical example — trivially "covered" by any forward call).

## Principled exclusions

The gap probe (`scripts/coverage-gap-probe.sh`) applies the following
exclusion list to FFI symbols. These paths do not need direct test
coverage; including them would dilute the signal.

| Symbol family | Why excluded | Effective coverage path |
|---|---|---|
| `tensor_print`, `tensor_live_count`, `tensor_peak_live_count` | Diagnostic-only; correctness has no operational impact | None |
| `backend_profile_reset`, `backend_profile_report`, `backend_reset_for_eval`, `backend_epoch_begin`, `backend_name` (and the `_return` RefC variants) | Lifecycle / profiling stubs; no semantic content | None |
| `get_rss_mb`, `get_current_rss_mb` | Memory introspection; OS-provided value | None |
| `mnist_load`, `mnist_count`, `mnist_get_image`, `mnist_get_label`, `mnist_free` | File-format-specific I/O; covered by `example-mnist` smoke gate | `make example-mnist` + `make test-examples` |
| `tensor_retain_handle`, `tensor_release_handle` | Refcount glue, covered implicitly by every tensor test | None |
| `idrisml_seq` | Pure pass-through sequencing helper | None |
| `tensor_mlx_compile_enabled`, `tensor_mlx_compile_invocations` | mlx-only diagnostic counters | None |
| Torch `at::*` / `torch::F::*` direct passthroughs (~80 of 93 `.cpp` files) | Coverage of `tensor_add` calling `torch::add` verifies plumbing, not kernel correctness; the kernel is libtorch's | Symbol coverage via the common Criterion suite (W3 tests run on torch too); custom-logic paths get W3b |
| Tape/mlx dispatch-table init (`backend_tape/training/autograd/op_dispatch.c`, `backend_mlx/training/autograd/op_dispatch.cpp`) | Pure initialization; trivially covered by any forward+backward | `test_op_dispatch.c` probes that `tape_dispatch_get(op) != NULL` for every OP_* |
| `Compatible` negative paths (F64-on-MPS, I64-on-MPS rejection) | Gated by type system at construction; cannot deterministically test "Metal rejects F64" without Metal hardware on every runner | Type-system enforcement + `check-rename-headers` CI gate |
| Multi-link unified-dispatch symbol forwarding | Tested implicitly by `example-transfer-demo` (multi-backend build) and the rename-headers CI gate | Existing example + CI gate |
| Dropout RNG output values | Inherently non-deterministic; no value oracle exists | Statistical mean test (already in suite) + example smoke. Per-element assertions are NOT allowed |
| mlx paravirt-GPU panic paths | Non-deterministic VM-level failure | Documented in `TODO.md` row 44 |
| `Data.Nat` recursive Peano walks | Performance footgun, not a correctness path | None |

## Categories of FFI symbol that the probe flags but exclusion is debatable

Some `extern "C"` symbols appear with zero test mentions but are
covered transitively through their parent. The probe lists them; the
intent is for `coverage-policy.md` to grow an explicit "transitive"
category when the population is stable. Examples observed today:

- **RefC-compat `_return` variants**: `param_register_return`,
  `tensor_backward_return_loss`, `tensor_to_doubles_return`, etc.
  These exist only for the RefC backend's value-threading FFI
  convention; their non-`_return` siblings are tested.
- **Dtype-suffixed creators**: `tensor_create_f32`, `tensor_create_f64`,
  `tensor_create_2d_f32`, `tensor_create_param_*_f{32,64}`. These
  resolve through the unsuffixed `tensor_create*` entry point which
  IS tested. Direct testing of every suffix would be redundant.

When a transitive-coverage symbol is genuinely uncovered (its
parent is also uncovered, or the parent's coverage doesn't actually
exercise the variant's code path), it goes into a real W3/W3b/W4
test, not into this exclusion table.

## Gap-probe usage

```bash
make coverage-gap-probe       # writes CSVs to build/, prints summary
```

Outputs:
- `build/coverage-gap-ops.csv` — per-OP_* status across tape + mlx
- `build/coverage-gap-symbols.csv` — per-FFI-symbol test-file hit count

Exit code is **advisory** (always 0) initially. Once W3+W4 close (the
OP_* gap is 0 on both backends) the script will be flipped to gating
in CI — adding a new OP_* tag without a corresponding test will fail
the build. See `.github/workflows/test.yml`.

## Coverage HTML reports (advisory, not a gate)

```bash
make coverage-backend-tape    # writes HTML to build-cov/html-tape/
make coverage-backend-torch   # ditto
make coverage-backend-mlx     # ditto
```

The HTML report is the place to spot per-line dead code or unreachable
branches. It is not used as a CI gate (the % can shift on benign
refactors that move code between files). Use it for one-shot
investigations, not regression tracking.

## Contributor checklist — when you add a new OP_*

When introducing a new `OP_FOO` to either tape or mlx:

1. **Forward FFI** — define `tensor_foo(...)` in the appropriate
   `backend_<b>/<area>/foo.{c,cpp}`.
2. **Backward** — register via `TAPE_REGISTER_OP(OP_FOO, ...)` (tape)
   or `MLX_REGISTER_REPLAY(OP_FOO, ...)` (mlx) in the same source
   file.
3. **Add a Criterion test** at `test/common/<area>/test_foo.c` that:
   - exercises `tensor_foo` forward with hand-computed expected
     values
   - calls `tensor_backward(loss)` and asserts gradients via
     `param_grad_item_at` against finite-difference or hand-computed
     reference
   - on torch passes the same test (the common-tree file runs on all
     three backends via `-DBACKEND_<b>` gating)
4. **TDD commit shape**: per `feedback_tdd_default`, either commit
   the test together with the implementing change (paired commit,
   body line `RED before this commit: <assertion>`) or commit the
   test under a skip flag first, then a follow-up commit removes
   the skip after the implementation lands.
5. **Run the probe**: `make coverage-gap-probe` should not list your
   new `OP_FOO` as MISSING.
6. **F32 routing**: if `OP_FOO` is routed through tape's F32
   storage path, also add a rung in the T29 ladder
   (`test_legacy_backend.c`). Skip-flag commit shape per the
   plan-doc's W5 section.
7. **Property-based test consideration**: ask whether `OP_FOO`
   has an invariant worth property-testing — round-trip,
   sum-to-one, norm-bounded, F32-vs-F64 oracle parity, idempotence,
   monotonicity. If yes, add a `prop_*` to
   `packages/idris-ml/src/Test/Properties/<Op>.idr` (see
   `Test/Properties/Softmax.idr` for a pure-Identity sum-to-one
   property and `Test/Properties/Reshape.idr` for a numel-preserved
   reshape invariant). If the invariant needs FFI-driven generators
   (i.e. the property body must construct real tensors via
   `tparam*` smart constructors), it's a `PropertyT IO ()` property
   — use the `checkPropertyIO` shim in
   `packages/idris-test/src/Test/Property.idr`. Not landing a
   property is fine; not *considering* one is the gap this checklist
   closes. The Axis 4 view above governs which kinds of properties
   land in the codebase.

## Contributor checklist — when you add a new FFI symbol

1. Declare in `packages/backends/backend.h`.
2. Implement in each backend (tape, torch, mlx).
3. Add a Criterion test that references the symbol by name (so the
   probe finds it). If the symbol is on the principled-exclusion list
   above, document why in the symbol's header doc comment.
4. Re-run `make coverage-gap-probe`; the symbol should not appear
   with `test_hits = 0`.

## Why we don't have a dedicated cross-backend agreement harness

The original coverage plan included **W6 — cross-backend numeric agreement test**:
a single multi-link Criterion file that would call tape's `tensor_add_tape`,
torch's `tensor_add_torch`, and mlx's `tensor_add_mlx` on identical input and
assert pairwise agreement within tolerance.

It is intentionally deferred. The same divergence-catching is achieved
by the common-test pattern at lower cost:

- The Criterion suite under `packages/backends/test/common/` is built **once
  per backend** (`-DBACKEND_<NAME>` selects assertions). When a test like
  `test_gelu.c::forward_backward_at_one` is added, it runs on tape, torch,
  and mlx in three separate CI invocations. If one backend's answer differs,
  that backend's CI lane fails; the other lanes stay green; you immediately
  know which backend disagrees.
- This is how `test_gelu.c` caught torch using exact GELU instead of the
  tanh approximation (commit `b23a090`), how `test_transpose_last2.c` caught
  mlx's `tensor_to_doubles` reading non-contiguous views in storage order
  (commit `5d5fc36`), and how `test_clip_grad_norm.c` captured torch's
  ~9e-8 post-clip precision drift (sentinel for TODO row 76).
- The dedicated W6 harness would add infrastructure cost (multi-link build
  variant, suffixed-FFI symbol declarations, cross-link `-DBACKEND_*` gating
  per file) for marginal additional coverage — the only case it captures
  that the common-test pattern doesn't is "all three forwards agree but
  backwards disagree across all three", which is a small target with the
  TODO row 76 sentinel already tracking the most-likely instance.

If a real cross-backend divergence appears that the common-test pattern
can't pin down (e.g. a multi-backend gradient-flow chain where the
problematic step isn't obvious), revisit W6 then. Until then, the common
tests are the gate.

## Cross-references

- `scripts/coverage-gap-probe.sh` — the probe itself
- `feedback_tdd_default` (in MEMORY.md) — TDD commit shapes
- `feedback_test_gates_must_run_in_ci` — CI must run the gates we
  build
- `feedback_memory_corruption_is_p1` — when a gap test surfaces a
  use-after-free or SIGBUS, file as P1 regardless of failure rate

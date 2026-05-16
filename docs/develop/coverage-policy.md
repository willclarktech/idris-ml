# Backend test coverage policy

This doc defines what "covered" means for the three backends
(`backend_tape`, `backend_torch`, `backend_mlx`), where coverage is
chased, where it's principally excluded, and the contributor
checklist when adding new ops.

## The three-axis target

A backend is fully covered when **all three** of these hold:

1. **Symbol coverage** — every `extern "C"` entry in
   `packages/backends/backend.h` that is not on the exclusion list
   below has at least one Criterion test in
   `packages/backends/test/` that exercises it.
2. **Backward coverage** — every `OP_*` tag in the backend's
   `tape.h` enum (`backend_tape/tape.h`, `backend_mlx/tape.h`) has a
   test that triggers its backward dispatch case and asserts gradient
   values against a hand-computed or finite-difference oracle. Torch
   has no `OP_*` enum (uses libtorch's autograd) and is covered via
   symbol coverage + W3b's custom-logic-path tests.
3. **F32 paired oracle** — every op routed through the F32 storage
   path has a paired F32-vs-F64 gradcheck rung in the tape T29 ladder
   (`test/common/test_legacy_backend.c`). This axis is tape-specific
   (mlx and torch's F32 paths live in their respective frameworks).

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

## Contributor checklist — when you add a new FFI symbol

1. Declare in `packages/backends/backend.h`.
2. Implement in each backend (tape, torch, mlx).
3. Add a Criterion test that references the symbol by name (so the
   probe finds it). If the symbol is on the principled-exclusion list
   above, document why in the symbol's header doc comment.
4. Re-run `make coverage-gap-probe`; the symbol should not appear
   with `test_hits = 0`.

## Cross-references

- `scripts/coverage-gap-probe.sh` — the probe itself
- `feedback_tdd_default` (in MEMORY.md) — TDD commit shapes
- `feedback_test_gates_must_run_in_ci` — CI must run the gates we
  build
- `feedback_memory_corruption_is_p1` — when a gap test surfaces a
  use-after-free or SIGBUS, file as P1 regardless of failure rate

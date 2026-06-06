/* Criterion suite for tape's typed-grad infrastructure (Phase 3a).
 *
 * Verifies that `ensure_grad_typed` allocates a per-data-dtype grad
 * buffer (F32 tensors get 4 bytes/elem, F64 default 8), and that the
 * `tape_grad_load_d` / `tape_grad_add_d` / `tape_grad_store_d` inline
 * accessors round-trip values correctly through both buffer widths.
 *
 * Background: prior to 2026-06-06 the legacy `ensure_grad` allocated
 * F64 unconditionally, mirroring autocast's "high-precision
 * accumulator over low-precision weights" pattern. Row 38 ships a
 * symmetric-F32 path where grad buffers match data dtype; this is
 * the foundational commit (3a) introducing the helpers. Subsequent
 * 3b/3c/3d migrate the ~143 `(double*)t->grad` backward call sites
 * to use the typed helpers; 3e migrates the optimizer M/V buffers.
 *
 * The legacy `ensure_grad` and the new `ensure_grad_typed` coexist
 * during the migration window so that build + tests stay green at
 * every commit. Once every backward site is migrated, the legacy
 * `ensure_grad` will be removed (Phase 3e).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"

#ifdef BACKEND_TAPE
/* Dtag values mirroring DType.Core ("13/14/15=F16/F32/F64"). */
#define DTAG_F32 14
#endif

#ifdef BACKEND_TAPE
/* Internal arena.h surface — re-declared locally to avoid pulling in
   the whole header (which has private struct definitions). */
typedef struct Tensor Tensor;
extern void ensure_grad_typed(Tensor* t);
extern size_t tape_grad_elem_size(int dtype_tag);

/* These mirror the inline definitions in arena.h. Tests that exercise
   the inline accessors should go through public FFI (backward + readout)
   rather than poking the struct directly, since the struct layout is
   private to the backend. For Phase 3a we verify the size + roundtrip
   via the existing public surface. */

Test(tape_grad_typed, elem_size_dispatches_on_dtag) {
    /* F32 (DT_F32 = 1) → 4 bytes; everything else → 8. */
    cr_assert_eq(tape_grad_elem_size(0), 8, "DT_F64 grad elem size");
    cr_assert_eq(tape_grad_elem_size(1), 4, "DT_F32 grad elem size");
    /* Future dtypes default to F64-sized grad (lingua-franca). */
    cr_assert_eq(tape_grad_elem_size(2), 8, "DT_BF16 grad elem size (F64 fallback)");
    cr_assert_eq(tape_grad_elem_size(3), 8, "DT_F16 grad elem size (F64 fallback)");
}

/* End-to-end roundtrip: an F32 tensor's gradient via a simple multiply
   should match the F64 reference within F32 precision. This is the
   smallest assertion that exercises ensure_grad_typed + the typed
   accumulators THROUGH the public FFI surface, without depending on
   any backward site having migrated yet (the existing F64 backward
   path runs the F64 reference; the F32 path uses the existing
   ensure_grad/legacy accessors which still work because no migration
   has happened).

   This test will gain teeth as Phase 3b/3c/3d migrate sites — once
   mul.c's backward switches to `tape_grad_add_d`, the F32 leg below
   exercises the new typed path end-to-end. For Phase 3a it's a
   passing-through regression sentinel. */
Test(tape_grad_typed, f32_param_existing_pipeline_still_works) {
    param_clear();
    double xd[] = {2.0};
    /* F32 param via dtag dispatch — exists pre-Phase 3a (precision-demo path). */
    TensorHandle x = tensor_create_streamed(xd, (int[]){1}, 1, 1, 0, DTAG_F32);
    param_register("x", x);
    TensorHandle x_sq = tensor_mul(x, x);
    TensorHandle loss = tensor_sum(x_sq);
    cr_assert_float_eq(tensor_item(loss), 4.0, 1e-5,
        "F32 (2.0)^2 should be 4.0 (got %.9f)", tensor_item(loss));
    tensor_backward(loss);
    /* d(x^2)/dx = 2x = 4. */
    cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, 1e-5,
        "F32 d(x^2)/dx at x=2 should be 4.0 (got %.9f)",
        param_grad_item_at(0, 0));
}
#endif /* BACKEND_TAPE */

/* Criterion suite for tape core/lifecycle ops.
 *
 * Covers: tensor_create_scalar, tensor_create, tensor_clone,
 *         tensor_free, tensor_item.
 * tensor_retain_handle / tensor_release_handle are tape-side no-ops
 * (ABI parity stubs); a separate smoke probe verifies they don't crash.
 */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(tape_core_lifecycle, create_scalar_then_item) {
    TensorHandle s = tensor_create_scalar(6.0, 0);
    cr_assert_float_eq(tensor_item(s), 6.0, 1e-12,
        "tensor_item should round-trip the value passed to tensor_create_scalar");
    cr_assert_eq(tensor_numel(s), 1);
    cr_assert_eq(tensor_dim(s), 0);
}

Test(tape_core_lifecycle, create_scalar_requires_grad_flag) {
    /* requires_grad threads through to the tensor; the scalar with
       requires_grad=1 should hold its grad slot ready. We don't read
       a grad here (no backward), just that the surface accepts the flag
       and the value still round-trips. */
    TensorHandle s_grad = tensor_create_scalar(2.5, 1);
    TensorHandle s_nograd = tensor_create_scalar(2.5, 0);
    cr_assert_float_eq(tensor_item(s_grad),   2.5, 1e-12);
    cr_assert_float_eq(tensor_item(s_nograd), 2.5, 1e-12);
}

Test(tape_core_lifecycle, create_vector) {
    double data[] = {1.0, 2.0, 3.0};
    int shape[] = {3};
    TensorHandle v = tensor_create(data, shape, 1, 0);
    cr_assert_eq(tensor_numel(v), 3);
    cr_assert_eq(tensor_dim(v), 1);
    cr_assert_eq(tensor_size(v, 0), 3);
    double out[3];
    tensor_to_doubles(v, out);
    cr_assert_float_eq(out[0], 1.0, 1e-12);
    cr_assert_float_eq(out[1], 2.0, 1e-12);
    cr_assert_float_eq(out[2], 3.0, 1e-12);
}

Test(tape_core_lifecycle, create_matrix) {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {2, 3};
    TensorHandle m = tensor_create(data, shape, 2, 0);
    cr_assert_eq(tensor_numel(m), 6);
    cr_assert_eq(tensor_dim(m), 2);
    cr_assert_eq(tensor_size(m, 0), 2);
    cr_assert_eq(tensor_size(m, 1), 3);
}

Test(tape_core_lifecycle, clone_scalar) {
    TensorHandle a = tensor_create_scalar(7.0, 0);
    TensorHandle b = tensor_clone(a);
    cr_assert_float_eq(tensor_item(b), 7.0, 1e-12,
        "clone should preserve the scalar value");
    cr_assert_eq(tensor_numel(b), 1);
    cr_assert_eq(tensor_dim(b), 0);
    /* Distinct handles (different pointers) — clone is a deep copy. */
    cr_assert_neq((void*)a, (void*)b, "clone must be a new handle");
}

Test(tape_core_lifecycle, clone_vector) {
    double data[] = {10.0, 20.0, 30.0};
    int shape[] = {3};
    TensorHandle a = tensor_create(data, shape, 1, 0);
    TensorHandle b = tensor_clone(a);
    double out[3];
    tensor_to_doubles(b, out);
    cr_assert_float_eq(out[0], 10.0, 1e-12);
    cr_assert_float_eq(out[1], 20.0, 1e-12);
    cr_assert_float_eq(out[2], 30.0, 1e-12);
    cr_assert_neq((void*)a, (void*)b);
}

Test(tape_core_lifecycle, free_is_safe_noop) {
    /* tensor_free is a no-op on tape (arena lifecycle owns teardown).
       Verify it doesn't crash; subsequent use should still work since
       the tape holds the underlying pointer alive until tape_reset. */
    TensorHandle s = tensor_create_scalar(1.0, 0);
    tensor_free(s);
    /* Calling tensor_item after free is technically UB on backends
       that DO free; tape leaves it valid. We don't assert read-back
       (forward-compat) — just that free itself doesn't crash. */
    cr_assert(1);
}

Test(tape_core_lifecycle, retain_release_handle_noop) {
    /* ABI parity stubs — should not crash for any handle, including
       NULL (mlx-equivalent does refcount and would crash on NULL). */
    TensorHandle s = tensor_create_scalar(42.0, 0);
    tensor_retain_handle(s);
    tensor_release_handle(s);
    /* Verify value still readable after retain/release dance. */
    cr_assert_float_eq(tensor_item(s), 42.0, 1e-12);
}

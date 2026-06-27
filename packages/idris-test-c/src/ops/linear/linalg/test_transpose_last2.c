/* Criterion suite for tensor_transpose_last2 (forward + backward).
 *
 *   [B, m, n] -> [B, n, m] swapping the last two axes per batch.
 *
 * Closes the W3 OP_TRANSPOSE_LAST2 coverage gap on tape. mlx and torch
 * both implement tensor_transpose_last2 too; this exercises all three.
 *
 * Reads via tensor_to_doubles (flat-buffer) so we don't depend on
 * tensor_item_3d (no such FFI).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(linear_linalg_transpose_last2, forward_single_batch) {
	/* x = [[[1, 2, 3], [4, 5, 6]]] shape [1, 2, 3]
	 * -> [[[1, 4], [2, 5], [3, 6]]] shape [1, 3, 2]
	 * Flat input order: 1,2,3,4,5,6.
	 * Flat output order: 1,4,2,5,3,6. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle flat = tensor_create_param_2d_f64(2, 3, hcopy(xd, 6));
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 2, 3);
	TensorHandle y = tensor_transpose_last2(x);
	double buf[6];
	tensor_to_doubles(y, buf);
	double expected[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], expected[i], TEST_TOL_RELAXED,
		                   "transpose_last2 flat[%d] should be %.1f (got %.9f)", i, expected[i],
		                   buf[i]);
	}
}

Test(linear_linalg_transpose_last2, double_transpose_is_identity) {
	/* transpose_last2(transpose_last2(x)) == x. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle flat = tensor_create_param_2d_f64(2, 3, hcopy(xd, 6));
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 2, 3);
	TensorHandle y = tensor_transpose_last2(tensor_transpose_last2(x));
	double buf[6];
	tensor_to_doubles(y, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "double-transpose flat[%d] should be %.1f (got %.9f)", i, xd[i], buf[i]);
	}
}

Test(linear_linalg_transpose_last2, forward_f32) {
	/* F32 path: same values as `forward_single_batch`, exercises the
	 * DT_F32 branch (transpose_last2.c lines 20-26). reshape_3d propagates
	 * the F32 dtype_tag. Small integers fit exactly in f32; readback carries
	 * ~1e-6 error, so assert at an explicit 1e-5 tolerance. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle flat = tensor_create_param_2d_streamed(2, 3, hcopy(xd, 6), 0, 14);
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 2, 3);
	TensorHandle y = tensor_transpose_last2(x);
	cr_assert_str_eq(tensor_dtype_name(y), "F32",
	                 "transpose_last2 should propagate F32 tag (got %s)", tensor_dtype_name(y));
	double buf[6];
	tensor_to_doubles(y, buf);
	double expected[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], expected[i], 1e-5,
		                   "transpose_last2 F32 flat[%d] should be %.1f (got %.9f)", i, expected[i],
		                   buf[i]);
	}
}

Test(linear_linalg_transpose_last2, backward_passes_through) {
	/* For loss = sum(transpose_last2(x)) = sum(x) (transpose preserves
	 * elements), d loss / d x[i] = 1 for all i. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle flat = tensor_create_param_2d_f64(2, 3, hcopy(xd, 6));
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 2, 3);
	TensorHandle y = tensor_transpose_last2(x);
	TensorHandle loss = tensor_sum(y);
	cr_assert_float_eq(tensor_item(loss), 21.0, TEST_TOL_RELAXED,
	                   "sum(transpose(x)) should match sum(x) = 21 (got %.9f)", tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED,
		                   "grad x_flat[%d] for sum-loss should be 1 (got %.9f)", i,
		                   param_grad_item_at(0, i));
	}
}

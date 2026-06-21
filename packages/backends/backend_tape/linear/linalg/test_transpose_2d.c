/* Criterion suite for tensor_transpose_2d (forward + backward).
 *
 *   r = a^T where a=[m,n], r=[n,m]. Backward: transpose the grad back.
 *
 * tape/mlx/torch all implement tensor_transpose_2d; this exercises all
 * three via the common-test pattern. Reads via tensor_to_doubles so we
 * don't depend on tensor_item_2d storage order assumptions.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(linear_linalg_transpose_2d, forward) {
	/* a = [[1, 2, 3], [4, 5, 6]] shape [2, 3]
	 * a^T = [[1, 4], [2, 5], [3, 6]] shape [3, 2]
	 * Flat output order: 1, 4, 2, 5, 3, 6. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle a = tensor_create_param_2d_f64(2, 3, hcopy(ad, 6));
	param_register("a", a);
	TensorHandle r = tensor_transpose_2d(a);
	cr_assert_eq(tensor_dim(r), 2);
	cr_assert_eq(tensor_size(r, 0), 3);
	cr_assert_eq(tensor_size(r, 1), 2);
	double buf[6];
	tensor_to_doubles(r, buf);
	double expected[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(buf[i], expected[i], TEST_TOL_RELAXED,
		                   "transpose_2d flat[%d] should be %.1f (got %.9f)", i, expected[i],
		                   buf[i]);
}

Test(linear_linalg_transpose_2d, double_transpose_is_identity) {
	/* (a^T)^T == a. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle a = tensor_create_param_2d_f64(2, 3, hcopy(ad, 6));
	param_register("a", a);
	TensorHandle r = tensor_transpose_2d(tensor_transpose_2d(a));
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 3);
	double buf[6];
	tensor_to_doubles(r, buf);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(buf[i], ad[i], TEST_TOL_RELAXED,
		                   "double-transpose flat[%d] should be %.1f (got %.9f)", i, ad[i], buf[i]);
}

Test(linear_linalg_transpose_2d, forward_f32) {
	/* F32 path: same values as `forward`, exercises the DT_F32 branch
	 * (transpose_2d.c lines 19-24). Values are small integers that fit
	 * exactly in f32; readback via tensor_to_doubles carries ~1e-6 error,
	 * so assert at an explicit 1e-5 tolerance (NOT TEST_TOL_TIGHT). */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle a = tensor_create_param_2d_streamed(2, 3, hcopy(ad, 6), 0, 14);
	param_register("a", a);
	TensorHandle r = tensor_transpose_2d(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "transpose_2d should propagate F32 tag (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_dim(r), 2);
	cr_assert_eq(tensor_size(r, 0), 3);
	cr_assert_eq(tensor_size(r, 1), 2);
	double buf[6];
	tensor_to_doubles(r, buf);
	double expected[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(buf[i], expected[i], 1e-5,
		                   "transpose_2d F32 flat[%d] should be %.1f (got %.9f)", i, expected[i],
		                   buf[i]);
}

Test(linear_linalg_transpose_2d, backward_transposes_grad) {
	/* loss = sum(a^T) = sum(a); d loss / d a[i] = 1 for all i.
	 * Exercises tape_backward_transpose_2d (transpose the grad back). */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle a = tensor_create_param_2d_f64(2, 3, hcopy(ad, 6));
	param_register("a", a);
	TensorHandle r = tensor_transpose_2d(a);
	TensorHandle loss = tensor_sum(r);
	cr_assert_float_eq(tensor_item(loss), 21.0, TEST_TOL_RELAXED,
	                   "sum(a^T) should match sum(a) = 21 (got %.9f)", tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED,
		                   "grad a[%d] for sum-loss should be 1 (got %.9f)", i,
		                   param_grad_item_at(0, i));
}

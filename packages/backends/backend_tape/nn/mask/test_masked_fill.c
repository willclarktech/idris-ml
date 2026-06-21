/* Criterion suite for tensor_masked_fill (forward + backward).
 *
 *   r[i] = mask[i] != 0 ? value : t[i]
 *   d r[i] / d t[i] = mask[i] == 0 ? 1 : 0   (value is a scalar, not a tensor)
 *
 * Closes the W3/W4 OP_MASKED_FILL coverage gap on tape + mlx.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

static double* heap_copy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

Test(nn_mask_masked_fill, forward_partial_mask) {
	/* t = [[1, 2, 3]], mask = [[0, 1, 0]], value = -1 -> [[1, -1, 3]]. */
	param_clear();
	double td[] = {1.0, 2.0, 3.0};
	double md[] = {0.0, 1.0, 0.0};
	TensorHandle t = tensor_create_param_2d_f64(1, 3, heap_copy(td, 3));
	TensorHandle mask = tensor_create_2d_f64(1, 3, heap_copy(md, 3), 0);
	param_register("t", t);
	TensorHandle r = tensor_masked_fill(t, mask, -1.0);
	cr_assert_float_eq(tensor_item_2d(r, 0, 0), 1.0, TEST_TOL_RELAXED,
	                   "masked_fill: unmasked position should keep original (got %.9f)",
	                   tensor_item_2d(r, 0, 0));
	cr_assert_float_eq(tensor_item_2d(r, 0, 1), -1.0, TEST_TOL_RELAXED,
	                   "masked_fill: masked position should be -1 (got %.9f)",
	                   tensor_item_2d(r, 0, 1));
	cr_assert_float_eq(tensor_item_2d(r, 0, 2), 3.0, TEST_TOL_RELAXED,
	                   "masked_fill: unmasked position should keep original (got %.9f)",
	                   tensor_item_2d(r, 0, 2));
}

Test(nn_mask_masked_fill, backward_pass_through_unmasked) {
	/* loss = sum(masked_fill(t, [0,1,0], -1)) = t[0] + (-1) + t[2] = 1 + (-1) + 3 = 3
	 * d loss / d t[0] = 1  (unmasked)
	 * d loss / d t[1] = 0  (masked — value is a constant, gradient is killed)
	 * d loss / d t[2] = 1  (unmasked) */
	param_clear();
	double td[] = {1.0, 2.0, 3.0};
	double md[] = {0.0, 1.0, 0.0};
	TensorHandle t = tensor_create_param_2d_f64(1, 3, heap_copy(td, 3));
	TensorHandle mask = tensor_create_2d_f64(1, 3, heap_copy(md, 3), 0);
	param_register("t", t);
	TensorHandle r = tensor_masked_fill(t, mask, -1.0);
	TensorHandle loss = tensor_sum(r);
	cr_assert_float_eq(tensor_item(loss), 3.0, TEST_TOL_RELAXED,
	                   "loss after masked_fill should be 3.0 (got %.9f)", tensor_item(loss));
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_RELAXED,
	                   "grad t[0,0] should pass through (got %.9f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, TEST_TOL_RELAXED,
	                   "grad t[0,1] should be zero where masked (got %.9f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, TEST_TOL_RELAXED,
	                   "grad t[0,2] should pass through (got %.9f)", param_grad_item_at(0, 2));
}

Test(nn_mask_masked_fill, all_masked_forward_and_backward) {
	/* mask = all 1s -> every element becomes `value`; gradient is killed
	 * everywhere (value is a constant). Exercises the all-true mask branch. */
	param_clear();
	double td[] = {1.0, 2.0, 3.0, 4.0};
	double md[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle t = tensor_create_param_2d_f64(2, 2, heap_copy(td, 4));
	TensorHandle mask = tensor_create_2d_f64(2, 2, heap_copy(md, 4), 0);
	param_register("t", t);
	TensorHandle r = tensor_masked_fill(t, mask, 7.0);
	double buf[4];
	tensor_to_doubles(r, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], 7.0, TEST_TOL_RELAXED,
		                   "all-masked r[%d] should be 7.0 (got %.9f)", i, buf[i]);
	TensorHandle loss = tensor_sum(r);
	cr_assert_float_eq(tensor_item(loss), 28.0, TEST_TOL_RELAXED,
	                   "all-masked loss should be 4*7=28 (got %.9f)", tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, TEST_TOL_RELAXED,
		                   "all-masked grad[%d] should be 0 (got %.9f)", i,
		                   param_grad_item_at(0, i));
}

Test(nn_mask_masked_fill, no_masked_is_identity_with_full_grad) {
	/* mask = all 0s -> result equals input; gradient flows fully through.
	 * Exercises the all-false mask branch. */
	param_clear();
	double td[] = {5.0, 6.0, 7.0};
	double md[] = {0.0, 0.0, 0.0};
	TensorHandle t = tensor_create_param_2d_f64(1, 3, heap_copy(td, 3));
	TensorHandle mask = tensor_create_2d_f64(1, 3, heap_copy(md, 3), 0);
	param_register("t", t);
	TensorHandle r = tensor_masked_fill(t, mask, -99.0);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(tensor_item_2d(r, 0, i), td[i], TEST_TOL_RELAXED,
		                   "no-mask r[%d] should keep original (got %.9f)", i,
		                   tensor_item_2d(r, 0, i));
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED,
		                   "no-mask grad[%d] should be 1.0 (got %.9f)", i,
		                   param_grad_item_at(0, i));
}

Test(nn_mask_masked_fill, no_grad_input_skips_tape) {
	/* requires_grad=0 input -> result has requires_grad=0 and no tape entry is
	 * appended (the `if (r->requires_grad)` branch is skipped). Forward still
	 * fills correctly. */
	param_clear();
	double td[] = {1.0, 2.0, 3.0};
	double md[] = {0.0, 1.0, 1.0};
	TensorHandle t = tensor_create_2d_f64(1, 3, heap_copy(td, 3), 0);
	TensorHandle mask = tensor_create_2d_f64(1, 3, heap_copy(md, 3), 0);
	TensorHandle r = tensor_masked_fill(t, mask, 0.0);
	cr_assert_float_eq(tensor_item_2d(r, 0, 0), 1.0, TEST_TOL_RELAXED,
	                   "no-grad fill unmasked (got %.9f)", tensor_item_2d(r, 0, 0));
	cr_assert_float_eq(tensor_item_2d(r, 0, 1), 0.0, TEST_TOL_RELAXED,
	                   "no-grad fill masked (got %.9f)", tensor_item_2d(r, 0, 1));
	cr_assert_float_eq(tensor_item_2d(r, 0, 2), 0.0, TEST_TOL_RELAXED,
	                   "no-grad fill masked (got %.9f)", tensor_item_2d(r, 0, 2));
}

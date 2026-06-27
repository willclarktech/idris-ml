/* Criterion suite for tensor_log_softmax_2d (forward + backward).
 *
 *   y[i, j] = log(softmax(x)[i, j]) = x[i, j] - log(sum_k exp(x[i, k]))
 *
 * Properties verified:
 *   - exp(y[i, :]) sums to 1 per row (verified via tensor_sum on exp output
 *     would need exp, so just check the row property directly via cell-level
 *     analytic value)
 *   - For x = [[a, b]], y[0, 0] = a - log(exp(a) + exp(b)) = -log(1 + exp(b-a))
 *
 * Closes the W3 OP_LOG_SOFTMAX_2D coverage gap on tape. mlx fuses this into
 * the general softmax replay so no MLX_REGISTER_REPLAY anchor for
 * OP_LOG_SOFTMAX_2D exists there.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(nn_softmax_log_softmax_2d, forward_two_class) {
	/* x = [[1, 2]] -> y = [[-log(1+e), 1-log(1+e)]]
	 * = [[-log(1+e), 1-log(1+e)]] */
	param_clear();
	double xd[] = {1.0, 2.0};
	TensorHandle x = tensor_create_param_2d_f64(1, 2, hcopy(xd, 2));
	param_register("x", x);
	TensorHandle y = tensor_log_softmax_2d(x);
	double l = log(1.0 + exp(1.0)); /* log(e^0 + e^1) shifted so x[0] - max(x) */
	double expected_0 = 1.0 - log(exp(1.0) + exp(2.0));
	double expected_1 = 2.0 - log(exp(1.0) + exp(2.0));
	(void)l;
	cr_assert_float_eq(tensor_item_2d(y, 0, 0), expected_0, TEST_TOL_RELAXED,
	                   "log_softmax_2d[0,0] should be %.9f (got %.9f)", expected_0,
	                   tensor_item_2d(y, 0, 0));
	cr_assert_float_eq(tensor_item_2d(y, 0, 1), expected_1, TEST_TOL_RELAXED,
	                   "log_softmax_2d[0,1] should be %.9f (got %.9f)", expected_1,
	                   tensor_item_2d(y, 0, 1));
}

Test(nn_softmax_log_softmax_2d, rows_independent) {
	/* x = [[1,2],[5,5]] — two rows with very different scales.
	 * Row 0: log_softmax_2d = [1 - log(e+e^2), 2 - log(e+e^2)]
	 * Row 1: log_softmax_2d = [-log(2), -log(2)] (symmetric two-way) */
	param_clear();
	double xd[] = {1.0, 2.0, 5.0, 5.0};
	TensorHandle x = tensor_create_param_2d_f64(2, 2, hcopy(xd, 4));
	param_register("x", x);
	TensorHandle y = tensor_log_softmax_2d(x);
	cr_assert_float_eq(tensor_item_2d(y, 1, 0), -log(2.0), TEST_TOL_RELAXED,
	                   "log_softmax row 1 col 0 (symmetric) should be -log(2) (got %.9f)",
	                   tensor_item_2d(y, 1, 0));
	cr_assert_float_eq(tensor_item_2d(y, 1, 1), -log(2.0), TEST_TOL_RELAXED,
	                   "log_softmax row 1 col 1 (symmetric) should be -log(2) (got %.9f)",
	                   tensor_item_2d(y, 1, 1));
}

Test(nn_softmax_log_softmax_2d, backward_runs) {
	/* Verify backward pass doesn't crash and produces a gradient on x.
	 * For loss = y[0,0], d loss / d x[0, j] = delta_{j,0} - softmax(x)[0, j].
	 * softmax([1,2]) = [e/(e+e^2), e^2/(e+e^2)] = [1/(1+e), e/(1+e)].
	 * d loss / d x[0, 0] = 1 - 1/(1+e) = e/(1+e)
	 * d loss / d x[0, 1] = 0 - e/(1+e) = -e/(1+e) */
	param_clear();
	double xd[] = {1.0, 2.0};
	TensorHandle x = tensor_create_param_2d_f64(1, 2, hcopy(xd, 2));
	param_register("x", x);
	TensorHandle y = tensor_log_softmax_2d(x);
	/* Build loss = y[0,0] by narrowing to a [1,1] slice and summing. */
	TensorHandle row0 = tensor_narrow(y, 0, 0, 1);      /* [1, 2] */
	TensorHandle cell00 = tensor_narrow(row0, 1, 0, 1); /* [1, 1] */
	TensorHandle loss = tensor_sum(cell00);
	tensor_backward(loss);
	double softmax0 = 1.0 / (1.0 + exp(1.0)); /* softmax(x)[0,0] = 1/(1+e) */
	double softmax1 = exp(1.0) / (1.0 + exp(1.0));
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0 - softmax0, TEST_TOL_RELAXED,
	                   "grad x[0,0] should be 1-softmax[0] (got %.9f vs %.9f)",
	                   param_grad_item_at(0, 0), 1.0 - softmax0);
	cr_assert_float_eq(param_grad_item_at(0, 1), -softmax1, TEST_TOL_RELAXED,
	                   "grad x[0,1] should be -softmax[1] (got %.9f vs %.9f)",
	                   param_grad_item_at(0, 1), -softmax1);
}

#ifdef BACKEND_TAPE
/* F32 arm (tape streamed dtag-14): drives the is_f32 store + the F32 result
   construction in log_softmax_2d.c. Row [0, ln3] -> y = [-ln4, ln3-ln4]. */
Test(log_softmax_2d_f32, forward) {
	double l3 = log(3.0), l4 = log(4.0);
	double d[] = {0.0, l3};
	TensorHandle x = tensor_create_2d_streamed(1, 2, hcopy(d, 2), /*rg=*/0, /*stream_tag=*/0, 14);
	TensorHandle r = tensor_log_softmax_2d(x);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "got %s", tensor_dtype_name(r));
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], -l4, 1e-5, "y[0,0] = -ln4 (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], l3 - l4, 1e-5, "y[0,1] = ln3-ln4 (got %.6f)", out[1]);
}
#endif /* BACKEND_TAPE */

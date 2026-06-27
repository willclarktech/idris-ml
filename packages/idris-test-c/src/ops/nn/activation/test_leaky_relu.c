/* Criterion suites for tensor_leaky_relu (forward + backward).
 *
 *   leaky_relu(x) = max(alpha*x, x) = (x >= 0) ? x : alpha*x
 *   d/dx          = (x >= 0) ? 1 : alpha
 *
 * Covers the F64 scalar (numel==1) and vector branches plus the
 * OP_LEAKY_RELU backward, and (under BACKEND_TAPE) the F32 scalar arm of
 * tensor_leaky_relu_f32 via the streamed dtag-14 creators.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

/* Scalar (numel==1) forward, positive input passes through unchanged. */
Test(nn_activation_leaky_relu, scalar_positive_forward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.5, 0);
	TensorHandle r = tensor_leaky_relu(a, 0.01);
	cr_assert_float_eq(tensor_item(r), 2.5, TEST_TOL_TIGHT,
	                   "leaky_relu(2.5) should be 2.5 (got %.9f)", tensor_item(r));
}

/* Scalar (numel==1) forward, negative input scaled by alpha. */
Test(nn_activation_leaky_relu, scalar_negative_forward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(-3.0, 0);
	TensorHandle r = tensor_leaky_relu(a, 0.1);
	cr_assert_float_eq(tensor_item(r), -0.3, TEST_TOL_TIGHT,
	                   "leaky_relu(-3.0, 0.1) should be -0.3 (got %.9f)", tensor_item(r));
}

/* Scalar backward: positive input -> grad 1. */
Test(nn_activation_leaky_relu, scalar_positive_backward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("a", a);
	TensorHandle loss = tensor_sum(tensor_leaky_relu(a, 0.01));
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_TIGHT,
	                   "d leaky_relu(4)/dx should be 1.0 (got %.9f)", param_grad_item_at(0, 0));
}

/* Scalar backward: negative input -> grad alpha. */
Test(nn_activation_leaky_relu, scalar_negative_backward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(-4.0, 1);
	param_register("a", a);
	TensorHandle loss = tensor_sum(tensor_leaky_relu(a, 0.2));
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.2, TEST_TOL_TIGHT,
	                   "d leaky_relu(-4, 0.2)/dx should be alpha=0.2 (got %.9f)",
	                   param_grad_item_at(0, 0));
}

/* Vector forward: mix of positive and negative, exactly-zero boundary (>=0 -> x). */
Test(nn_activation_leaky_relu, vector_forward) {
	param_clear();
	double d[] = {1.0, -1.0, 0.0, -2.5, 3.0};
	double alpha = 0.05;
	TensorHandle a = tensor_create_1d_f64(5, hcopy(d, 5), 0);
	TensorHandle r = tensor_leaky_relu(a, alpha);
	double buf[5];
	tensor_to_doubles(r, buf);
	for (int i = 0; i < 5; i++) {
		double expect = d[i] >= 0 ? d[i] : alpha * d[i];
		cr_assert_float_eq(buf[i], expect, TEST_TOL_TIGHT, "leaky_relu[%d] expected %.9f got %.9f",
		                   i, expect, buf[i]);
	}
}

#ifdef BACKEND_TAPE
/* Vector backward: per-element grad is 1 (x>=0) or alpha (x<0). loss = sum -> dout=1. */
Test(nn_activation_leaky_relu, vector_backward) {
	param_clear();
	double d[] = {1.0, -1.0, 0.0, -2.5, 3.0};
	double alpha = 0.05;
	TensorHandle a = tensor_create_1d_f64(5, hcopy(d, 5), 1);
	param_register("a", a);
	TensorHandle loss = tensor_sum(tensor_leaky_relu(a, alpha));
	tensor_backward(loss);
	for (int i = 0; i < 5; i++) {
		double expect = d[i] >= 0 ? 1.0 : alpha;
		cr_assert_float_eq(param_grad_item_at(0, i), expect, TEST_TOL_TIGHT,
		                   "d leaky_relu[%d]/dx expected %.9f got %.9f", i, expect,
		                   param_grad_item_at(0, i));
	}
}

/* alpha=0.25, x=-4 (F32 scalar). Negative leg: y = alpha*x = -1.
   loss=sum (scalar identity) -> grad_out=1, d_x = alpha = 0.25.
   Drives the F32 scalar arm + tape_append + the x<0 branch. */
Test(leaky_relu_cov, f32_scalar_negative) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(-4.0, 1, 0, 14);
	param_register("x", x);
	TensorHandle r = tensor_leaky_relu(x, 0.25);
	cr_assert_eq(tensor_numel(r), 1);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 in -> F32 out (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_float_eq(tensor_item_1d(r, 0), -1.0, TEST_TOL_RELAXED,
	                   "leaky_relu(-4, .25) should be -1 (got %.6f)", tensor_item_1d(r, 0));
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, TEST_TOL_RELAXED,
	                   "d_x should be alpha=0.25 (got %.6f)", param_grad_item_at(0, 0));
	param_clear();
}

/* alpha=0.25, x=4 (F32 scalar). Positive leg: y = x = 4.
   loss=sum -> grad_out=1, d_x = 1. Same F32 scalar arm, x>=0 branch. */
Test(leaky_relu_cov, f32_scalar_positive) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(4.0, 1, 0, 14);
	param_register("x", x);
	TensorHandle r = tensor_leaky_relu(x, 0.25);
	cr_assert_eq(tensor_numel(r), 1);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 in -> F32 out (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_float_eq(tensor_item_1d(r, 0), 4.0, TEST_TOL_RELAXED,
	                   "leaky_relu(4, .25) should be 4 (got %.6f)", tensor_item_1d(r, 0));
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_RELAXED,
	                   "d_x should be 1 (got %.6f)", param_grad_item_at(0, 0));
	param_clear();
}
#endif /* BACKEND_TAPE */

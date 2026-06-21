/* Criterion suite `leaky_relu_cov` — coverage top-up for tape leaky_relu.c.
 *
 * The pre-existing suite covers the F64 paths and the F32 multi-element arm.
 * This file closes the remaining uncovered arm: the F32 SCALAR (numel==1)
 * branch of tensor_leaky_relu_f32 (leaky_relu.c lines 17-20) — the
 * make_scalar_f32 fast path plus its tape_append. Both ternary legs
 * (x>=0 -> x, x<0 -> alpha*x) are exercised by two scalars.
 *
 * F32 tensors on tape must be built via the streamed dtag-14 creators
 * (bare *_f32 creators abort). All values are integer/quarter-exact in
 * single precision, so TEST_TOL_RELAXED is generous.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

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

/* Criterion suite for tape unary elementwise ops.
 * Covers neg, abs, exp, log, sqrt. */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

/* Heap-copy a stack array — the streamed creators take ownership and free it. */

Test(core_elementwise_neg, forward_backward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(7.0, 1);
	param_register("a", a);
	TensorHandle r = tensor_neg(a);
	cr_assert_float_eq(tensor_item(r), -7.0, 1e-12);
	tensor_backward(r);
	cr_assert_float_eq(param_grad_item_at(0, 0), -1.0, 1e-12, "d(-x)/dx should be -1 (got %.6f)",
	                   param_grad_item_at(0, 0));
}

Test(core_elementwise_abs, forward_backward_pos_neg) {
	param_clear();
	TensorHandle a = tensor_create_scalar(-3.0, 1);
	param_register("a", a);
	TensorHandle r = tensor_abs(a);
	cr_assert_float_eq(tensor_item(r), 3.0, 1e-12);
	tensor_backward(r);
	cr_assert_float_eq(param_grad_item_at(0, 0), -1.0, 1e-12,
	                   "d|x|/dx at x=-3 should be -1 (got %.6f)", param_grad_item_at(0, 0));
}

#ifdef BACKEND_TAPE
Test(core_elementwise_abs, backward_vector_mixed_signs_and_zero) {
	/* Exercises the multi-element backward loop and both sides of the
	   sign branch (>=0 -> +1, <0 -> -1). At x=0 the convention is +1. */
	param_clear();
	double ad[] = {-2.0, 0.0, 5.0};
	int s[] = {3};
	TensorHandle a = tensor_create(ad, s, 1, 1);
	param_register("a", a);
	TensorHandle r = tensor_abs(a);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 2.0, 1e-12);
	cr_assert_float_eq(out[1], 0.0, 1e-12);
	cr_assert_float_eq(out[2], 5.0, 1e-12);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), -1.0, 1e-12, "d|x|/dx at x=-2 should be -1");
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12, "d|x|/dx at x=0 should be +1");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "d|x|/dx at x=5 should be +1");
}
#endif /* BACKEND_TAPE */

Test(core_elementwise_exp, forward_backward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("a", a);
	TensorHandle r = tensor_exp(a);
	cr_assert_float_eq(tensor_item(r), exp(1.0), TEST_TOL_TIGHT);
	tensor_backward(r);
	/* d(exp(x))/dx at x=1 = exp(1) */
	cr_assert_float_eq(param_grad_item_at(0, 0), exp(1.0), TEST_TOL_TIGHT,
	                   "d(exp(x))/dx at x=1 should be e (got %.6f)", param_grad_item_at(0, 0));
}

Test(core_elementwise_log, forward_backward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("a", a);
	TensorHandle r = tensor_log(a);
	cr_assert_float_eq(tensor_item(r), log(2.0), TEST_TOL_TIGHT);
	tensor_backward(r);
	/* d(log(x))/dx at x=2 = 1/2 */
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, TEST_TOL_TIGHT,
	                   "d(log(x))/dx at x=2 should be 0.5 (got %.6f)", param_grad_item_at(0, 0));
}

Test(core_elementwise_sqrt, forward_backward) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("a", a);
	TensorHandle r = tensor_sqrt(a);
	cr_assert_float_eq(tensor_item(r), 2.0, TEST_TOL_TIGHT);
	tensor_backward(r);
	/* d(sqrt(x))/dx at x=4 = 1/(2*sqrt(4)) = 0.25 */
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, TEST_TOL_TIGHT,
	                   "d(sqrt(x))/dx at x=4 should be 0.25 (got %.6f)", param_grad_item_at(0, 0));
}

/* ---- F32 coverage: each unary op's fn_*_f32 + the F32 multi-element
   stamping of unop_elementwise (_kernels.inc vDSP/vForce switch arms).
   Multi-element (numel>1) inputs to bypass the scalar shortcut and hit the
   per-op vDSP/VV_* case. F32 readback carries ~1e-6 error -> tol 1e-5. */

/* Scalar (numel==1) F32 unary: the unop_elementwise scalar shortcut calls
   the scalar fn pointer directly (fn_neg_f32 / fn_exp_f32 / fn_sqrt_f32 /
   fn_abs_f32), which the multi-element vDSP path bypasses on Apple. */

Test(core_elementwise_neg, f32_scalar_fn) {
	double v = 7.0;
	TensorHandle a = tensor_create_scalar_streamed(v, 0, 0, 14);
	TensorHandle r = tensor_neg(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	cr_assert_float_eq(tensor_item(r), -7.0, 1e-5);
}

Test(core_elementwise_exp, f32_scalar_fn) {
	double v = 1.0;
	TensorHandle a = tensor_create_scalar_streamed(v, 0, 0, 14);
	TensorHandle r = tensor_exp(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	cr_assert_float_eq(tensor_item(r), exp(1.0), 1e-5);
}

Test(core_elementwise_sqrt, f32_scalar_fn) {
	double v = 4.0;
	TensorHandle a = tensor_create_scalar_streamed(v, 0, 0, 14);
	TensorHandle r = tensor_sqrt(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	cr_assert_float_eq(tensor_item(r), 2.0, 1e-5);
}

Test(core_elementwise_abs, f32_scalar_fn) {
	double v = -3.0;
	TensorHandle a = tensor_create_scalar_streamed(v, 0, 0, 14);
	TensorHandle r = tensor_abs(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	cr_assert_float_eq(tensor_item(r), 3.0, 1e-5);
}

#ifdef BACKEND_TAPE
Test(core_elementwise_neg, f32_forward_backward_vector) {
	param_clear();
	double ad[] = {1.0, -2.0, 3.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 14);
	param_register("a", a);
	TensorHandle r = tensor_neg(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 neg output keeps F32 tag");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], -1.0, 1e-5);
	cr_assert_float_eq(out[1], 2.0, 1e-5);
	cr_assert_float_eq(out[2], -3.0, 1e-5);
	tensor_backward(tensor_sum(r));
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), -1.0, 1e-5, "d(-x)/dx should be -1");
	param_clear();
}
#endif /* BACKEND_TAPE */

#ifdef BACKEND_TAPE
Test(core_elementwise_exp, f32_forward_backward_vector) {
	param_clear();
	double ad[] = {0.0, 1.0, 2.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 14);
	param_register("a", a);
	TensorHandle r = tensor_exp(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 1.0, 1e-5);
	cr_assert_float_eq(out[1], exp(1.0), 1e-5);
	cr_assert_float_eq(out[2], exp(2.0), 1e-4);
	tensor_backward(tensor_sum(r));
	/* d(exp(x))/dx = exp(x) */
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-5);
	cr_assert_float_eq(param_grad_item_at(0, 1), exp(1.0), 1e-5);
	param_clear();
}
#endif /* BACKEND_TAPE */

#ifdef BACKEND_TAPE
Test(core_elementwise_sqrt, f32_forward_backward_vector) {
	param_clear();
	double ad[] = {4.0, 9.0, 16.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 14);
	param_register("a", a);
	TensorHandle r = tensor_sqrt(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 2.0, 1e-5);
	cr_assert_float_eq(out[1], 3.0, 1e-5);
	cr_assert_float_eq(out[2], 4.0, 1e-5);
	tensor_backward(tensor_sum(r));
	/* d(sqrt(x))/dx at x=4 = 0.25, at x=9 = 1/6, at x=16 = 0.125 */
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, 1e-5);
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.125, 1e-5);
	param_clear();
}
#endif /* BACKEND_TAPE */

#ifdef BACKEND_TAPE
Test(core_elementwise_abs, f32_forward_backward_vector) {
	param_clear();
	double ad[] = {-2.0, 0.0, 5.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 14);
	param_register("a", a);
	TensorHandle r = tensor_abs(a);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 2.0, 1e-5);
	cr_assert_float_eq(out[1], 0.0, 1e-5);
	cr_assert_float_eq(out[2], 5.0, 1e-5);
	tensor_backward(tensor_sum(r));
	cr_assert_float_eq(param_grad_item_at(0, 0), -1.0, 1e-5, "d|x|/dx at x=-2 should be -1");
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-5, "d|x|/dx at x=0 should be +1");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-5, "d|x|/dx at x=5 should be +1");
	param_clear();
}
#endif /* BACKEND_TAPE */

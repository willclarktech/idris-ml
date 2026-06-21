/* Criterion suites for tape `tensor_div`.
 *
 * `core_elementwise_div` covers the F64 forward + backward happy path.
 * `div_cov` (tape-only) closes the F32 forward/backward arm via the streamed
 * dtag-14 creators. */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

Test(core_elementwise_div, forward_scalar) {
	TensorHandle a = tensor_create_scalar(10.0, 0);
	TensorHandle b = tensor_create_scalar(4.0, 0);
	TensorHandle c = tensor_div(a, b);
	cr_assert_float_eq(tensor_item(c), 2.5, 1e-12);
}

Test(core_elementwise_div, backward_scalar) {
	/* c = a/b; dc/da = 1/b, dc/db = -a/b^2 */
	param_clear();
	TensorHandle a = tensor_create_scalar(10.0, 1);
	TensorHandle b = tensor_create_scalar(4.0, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_div(a, b);
	tensor_backward(c);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, 1e-12,
	                   "d(a/b)/da should be 1/b=0.25 (got %.6f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(1, 0), -0.625, 1e-12,
	                   "d(a/b)/db should be -a/b^2=-0.625 (got %.6f)", param_grad_item_at(1, 0));
}

Test(core_elementwise_div, backward_vector) {
	param_clear();
	double ad[] = {6.0, 12.0};
	double bd[] = {2.0, 4.0};
	int s[] = {2};
	TensorHandle a = tensor_create(ad, s, 1, 1);
	TensorHandle b = tensor_create(bd, s, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_div(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	/* d(sum(a/b))/da[i] = 1/b[i], d(sum(a/b))/db[i] = -a[i]/b[i]^2 */
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0 / 2.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0 / 4.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(1, 0), -6.0 / (2.0 * 2.0), 1e-12);
	cr_assert_float_eq(param_grad_item_at(1, 1), -12.0 / (4.0 * 4.0), 1e-12);
}

#ifdef BACKEND_TAPE

/* a=[6,8,16] / b=[2,4,8] = [3,2,2]. Drives the F32 forward arm of
   tensor_div -> binop_elementwise_f32_disp -> fn_div_f32 (lines 21-22). */
Test(div_cov, f32_forward) {
	double ad[] = {6.0, 8.0, 16.0};
	double bd[] = {2.0, 4.0, 8.0};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(ad, 3), 0, 0, 14);
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(bd, 3), 0, 0, 14);
	TensorHandle r = tensor_div(a, b);
	cr_assert_eq(tensor_numel(r), 3);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	double out[3];
	tensor_to_doubles(r, out);
	double expected[] = {3.0, 2.0, 2.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED,
		                   "div_f32[%d] should be %.1f (got %.6f)", i, expected[i], out[i]);
}

/* Same F32 forward arm, plus backward to exercise the path end-to-end.
   d(a/b)/da = 1/b = [0.5, 0.25, 0.125]; d(a/b)/db = -a/b^2 =
   -[6/4, 8/16, 16/64] = -[1.5, 0.5, 0.25] (all power-of-two exact in F32). */
Test(div_cov, f32_forward_backward) {
	param_clear();
	double ad[] = {6.0, 8.0, 16.0};
	double bd[] = {2.0, 4.0, 8.0};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(ad, 3), 1, 0, 14);
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(bd, 3), 1, 0, 14);
	param_register("a", a);
	param_register("b", b);
	TensorHandle r = tensor_div(a, b);
	cr_assert_eq(tensor_numel(r), 3);
	double out[3];
	tensor_to_doubles(r, out);
	double expected[] = {3.0, 2.0, 2.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED,
		                   "div_f32[%d] should be %.1f (got %.6f)", i, expected[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expected_da[] = {0.5, 0.25, 0.125};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected_da[i], TEST_TOL_RELAXED,
		                   "da[%d] should be %.4f (got %.6f)", i, expected_da[i],
		                   param_grad_item_at(0, i));
	double expected_db[] = {-1.5, -0.5, -0.25};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), expected_db[i], TEST_TOL_RELAXED,
		                   "db[%d] should be %.4f (got %.6f)", i, expected_db[i],
		                   param_grad_item_at(1, i));
	param_clear();
}

#endif /* BACKEND_TAPE */

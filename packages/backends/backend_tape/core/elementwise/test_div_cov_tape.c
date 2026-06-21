/* Criterion suite `div_cov` — coverage top-up for tape core/elementwise/div.c.
 *
 * The pre-existing elementwise suites cover the F64 div forward+backward
 * happy path. This file closes the remaining uncovered F32 forward arm:
 *
 *   - div.c lines 21-22 : fn_div_f32 (the float a/b kernel), reached only
 *                         via the DT_F32 dispatch arm of tensor_div
 *                         (binop_elementwise_f32_disp, gated by the
 *                         dtype_tag == DT_F32 check). Entering that arm with
 *                         matching F32 tags also evaluates the line-29
 *                         mixed-dtype guard condition (false, no abort).
 *
 * Tape forbids bare f32 creators (they abort), so F32 tensors are built via
 * the streamed dtag-14 creators (they own + free their hcopy buffer). All
 * values are integer / power-of-two exact in single precision, so reads use
 * TEST_TOL_RELAXED and the hand-computed oracles are exact.
 *
 * Tape-only: the streamed dtag-14 path + F32-on-tape oracle are tape
 * specifics; the file compiles into every backend's test binary, so the
 * whole suite is guarded by BACKEND_TAPE.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

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

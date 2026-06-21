/* Criterion suite `pow_cov` — coverage top-up for tape core/elementwise/pow.c.
 *
 * The base pow coverage exercises the F64 forward + backward arms. This file
 * closes the remaining uncovered arm:
 *
 *   - pow.c:21-22 (fn_pow_f32 / powf) — the F32 forward kernel, reached only
 *     when tensor_pow sees an F32-tagged input and dispatches through
 *     binop_elementwise_f32_disp(..., OP_POW, fn_pow_f32).
 *
 * F32 tensors on tape MUST be built via the streamed dtag-14 creators (bare
 * tensor_create_*_f32 aborts). Oracles are hand-computed; all chosen base/
 * exponent values are small integers whose powers are exact in single
 * precision, so F32 readback/grad assertions use TEST_TOL_RELAXED generously.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* a=[2,3,4] (F32), b=[2,2,2] (F32). a^b = [4,9,16].
   Hits fn_pow_f32 via the F32 forward dispatch in tensor_pow. */
Test(pow_cov, pow_f32_forward) {
	double ad[] = {2.0, 3.0, 4.0};
	double bd[] = {2.0, 2.0, 2.0};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(ad, 3), 0, 0, 14);
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(bd, 3), 0, 0, 14);
	TensorHandle r = tensor_pow(a, b);
	cr_assert_eq(tensor_numel(r), 3);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	double out[3];
	tensor_to_doubles(r, out);
	double expected[] = {4.0, 9.0, 16.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED,
		                   "pow_f32[%d] should be %.1f (got %.6f)", i, expected[i], out[i]);
}

/* Same F32 forward arm, with grad-tracking params so the result feeds a sum
   loss + backward. d(a^b)/da = b*a^(b-1) = 2*a = [4,6,8] (integer-exact).
   Confirms the F32 forward kernel composes with the (F64-accumulated)
   backward; only the integer-exact da arm is asserted. */
Test(pow_cov, pow_f32_forward_backward_da) {
	param_clear();
	double ad[] = {2.0, 3.0, 4.0};
	double bd[] = {2.0, 2.0, 2.0};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(ad, 3), 1, 0, 14);
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(bd, 3), 1, 0, 14);
	param_register("a", a);
	param_register("b", b);
	TensorHandle r = tensor_pow(a, b);
	cr_assert_eq(tensor_numel(r), 3);
	double out[3];
	tensor_to_doubles(r, out);
	double expected[] = {4.0, 9.0, 16.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED,
		                   "pow_f32[%d] should be %.1f (got %.6f)", i, expected[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expected_da[] = {4.0, 6.0, 8.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected_da[i], TEST_TOL_RELAXED,
		                   "da[%d] should be %.1f (got %.6f)", i, expected_da[i],
		                   param_grad_item_at(0, i));
	param_clear();
}

#endif /* BACKEND_TAPE */

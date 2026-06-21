/* Coverage suite for tape single-sample `tensor_conv2d` F32 arms.
 *
 * The base coverage suite (test_conv2d_cov_tape.c) exercises only the F64
 * storage path. This file closes the streamed-dtag-14 (F32) forward arms of
 * tensor_conv2d that the F64 tests never reach:
 *   - the is_f32 arena_alloc output buffer + the (float)val store branch
 *     (conv2d.c:40, 59-60);
 *   - the make_tensor_arena_f32 result construction (conv2d.c:68-69).
 *
 * The backward op (tape_backward_conv2d) is dtype-agnostic (it reads via
 * tape_load_d / tape_grad_load_d, which widen F32 to double), so we also
 * drive it on the F32 tensors and assert the gradients to confirm the F32
 * forward feeds a correct tape.
 *
 * NOT covered here: conv2d.c:30 (the mixed-dtype abort guard) — abort guards
 * are death tests excluded per the suite policy.
 *
 * All oracles are hand-computed; every value is an integer exactly
 * representable in single precision, so the F32 store + widen round-trips
 * exactly and reads use TEST_TOL_RELAXED.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* inC=1, H=2, W=2, outC=1, kH=2, kW=2, pad=0, stride=1 -> oH=oW=1.
   Single output window. F32 input/kernel/bias via streamed dtag 14.
   Drives the is_f32 forward store (conv2d.c:59-60) +
   make_tensor_arena_f32 (conv2d.c:68-69). */
Test(conv2d_f32_cov, f32_forward_backward_bias) {
	param_clear();
	double in_src[4] = {1.0, 2.0, 3.0, 4.0};
	double k_src[4] = {1.0, 1.0, 1.0, 1.0};
	double b_src[1] = {0.5};
	int sh_in[3] = {1, 2, 2};
	int sh_k[4] = {1, 1, 2, 2};

	TensorHandle in = tensor_create_streamed(hcopy(in_src, 4), sh_in, 3, 1, 0, 14);
	TensorHandle k = tensor_create_streamed(hcopy(k_src, 4), sh_k, 4, 1, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(1, hcopy(b_src, 1), 1, 0, 14);
	param_register("in", in);
	param_register("k", k);
	param_register("bias", bias);

	TensorHandle out = tensor_conv2d(in, k, bias, 0, 0, 1, 1);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	/* out = bias + sum(in*k) = 0.5 + (1+2+3+4) = 10.5 */
	cr_assert_float_eq(tensor_item_1d(out, 0), 10.5, TEST_TOL_RELAXED, "out[0] should be 10.5");

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* d_in[i] = kernel weight = 1 (single window, sum loss). */
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED, "d_in[%d]", i);
	/* d_k[pos] = input at that window position. */
	double exp_k[4] = {1.0, 2.0, 3.0, 4.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], TEST_TOL_RELAXED, "d_k[%d]", i);
	/* d_bias = number of output positions = 1. */
	cr_assert_float_eq(param_grad_item_at(2, 0), 1.0, TEST_TOL_RELAXED, "d_bias");

	param_clear();
}

/* inC=1, H=3, W=3, outC=1, kH=2, kW=2, pad=0, stride=1 -> oH=oW=2.
   Multi-window F32 forward: exercises the is_f32 store across all four
   output positions (conv2d.c:59-60) and a non-trivial overlapping
   backward, still integer-exact in F32. */
Test(conv2d_f32_cov, f32_forward_multiwindow) {
	param_clear();
	double in_src[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	double k_src[4] = {1.0, 1.0, 1.0, 1.0};
	double b_src[1] = {0.0};
	int sh_in[3] = {1, 3, 3};
	int sh_k[4] = {1, 1, 2, 2};

	TensorHandle in = tensor_create_streamed(hcopy(in_src, 9), sh_in, 3, 1, 0, 14);
	TensorHandle k = tensor_create_streamed(hcopy(k_src, 4), sh_k, 4, 1, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(1, hcopy(b_src, 1), 1, 0, 14);
	param_register("in", in);
	param_register("k", k);
	param_register("bias", bias);

	TensorHandle out = tensor_conv2d(in, k, bias, 0, 0, 1, 1);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	/* 2x2 windows over a 3x3 input, kernel all-ones, no bias:
	   out[0,0]=1+2+4+5=12, out[0,1]=2+3+5+6=16,
	   out[1,0]=4+5+7+8=24, out[1,1]=5+6+8+9=28 */
	double exp_out[4] = {12.0, 16.0, 24.0, 28.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(tensor_item_1d(out, i), exp_out[i], TEST_TOL_RELAXED, "out[%d]", i);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* d_in = window-membership count per position (sum loss, kernel ones). */
	double exp_in[9] = {1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0};
	for (int i = 0; i < 9; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), exp_in[i], TEST_TOL_RELAXED, "d_in[%d]", i);
	/* d_k[pos] = sum over windows of input at that window position. */
	double exp_k[4] = {12.0, 16.0, 24.0, 28.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], TEST_TOL_RELAXED, "d_k[%d]", i);
	/* d_bias = number of output positions = 4. */
	cr_assert_float_eq(param_grad_item_at(2, 0), 4.0, TEST_TOL_RELAXED, "d_bias");

	param_clear();
}

#endif /* BACKEND_TAPE */

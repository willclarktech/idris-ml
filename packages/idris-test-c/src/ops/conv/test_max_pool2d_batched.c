/* Criterion suites for tape `tensor_max_pool2d_batched`.
 *
 * Covers the F64 forward/backward path plus the streamed F32 (dtag-14)
 * forward/backward and no-grad index-free arms.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

/* Dtag value mirroring DType.Core ("14=F32"). */
#define DTAG_F32 14

/* input [B=2, C=1, H=2, W=2] = [[[1,2],[3,4]], [[8,7],[6,5]]], k=2 s=1.
 * Forward winners: [4, 8]. Backward sum-loss: d_in winner positions = 1.
 *   sample 0 winner: in[3] (=4)
 *   sample 1 winner: in[4] (=8)
 */
Test(conv_max_pool2d_batched, forward_and_backward) {
	param_clear();
	double in_data[8] = {1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0};
	int sh[4] = {2, 1, 2, 2};
	TensorHandle in = tensor_create(in_data, sh, 4, 1);
	param_register("in", in);

	TensorHandle out = tensor_max_pool2d_batched(in, 2, 2, 1, 1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 4.0, 1e-12);
	cr_assert_float_eq(tensor_item_1d(out, 1), 8.0, 1e-12);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	/* in[3]=4 (sample 0 winner), in[4]=8 (sample 1 winner) */
	double expected[8] = {0, 0, 0, 1, 1, 0, 0, 0};
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected[i], 1e-12, "d_in[%d]", i);
}

#ifdef BACKEND_TAPE
/* B=1, C=1, H=4, W=4, kH=kW=2, stride=2 -> oH=oW=2 (4 non-overlapping
   windows). F32 input WITH requires_grad: drives the F32 forward store
   (line 54), the make_tensor_arena_f32 result (line 65), and the backward
   scatter through max_indices. Input is 1..16 row-major so each window's
   max sits at its bottom-right corner. */
Test(max_pool2d_batched_cov, f32_forward_backward) {
	param_clear();
	double in_src[16];
	for (int i = 0; i < 16; i++)
		in_src[i] = (double)(i + 1);
	int sh_in[4] = {1, 1, 4, 4};

	TensorHandle in = tensor_create_streamed(hcopy(in_src, 16), sh_in, 4, 1, 0, DTAG_F32);
	param_register("in", in);

	TensorHandle out = tensor_max_pool2d_batched(in, 2, 2, 2, 2);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	/* out order = oh*oW+ow; window maxes are bottom-right of each 2x2 block. */
	cr_assert_float_eq(tensor_item_1d(out, 0), 6.0, TEST_TOL_RELAXED, "out[0]");
	cr_assert_float_eq(tensor_item_1d(out, 1), 8.0, TEST_TOL_RELAXED, "out[1]");
	cr_assert_float_eq(tensor_item_1d(out, 2), 14.0, TEST_TOL_RELAXED, "out[2]");
	cr_assert_float_eq(tensor_item_1d(out, 3), 16.0, TEST_TOL_RELAXED, "out[3]");

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* Sum-loss: grad 1 lands on each window's argmax input position
	   (flat indices 5, 7, 13, 15); everywhere else 0. */
	double exp_in[16] = {0};
	exp_in[5] = exp_in[7] = exp_in[13] = exp_in[15] = 1.0;
	for (int i = 0; i < 16; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), exp_in[i], TEST_TOL_RELAXED, "d_in[%d]", i);

	param_clear();
}

/* B=1, C=1, H=2, W=2, kH=kW=2, stride=2 -> oH=oW=1. F32 input WITHOUT
   requires_grad: no tape entry is appended, so the else-branch frees
   max_idx (line 87). Single window -> max of {1,2,3,4} = 4. */
Test(max_pool2d_batched_cov, f32_no_grad_frees_indices) {
	param_clear();
	double in_src[4] = {1.0, 2.0, 3.0, 4.0};
	int sh_in[4] = {1, 1, 2, 2};

	TensorHandle in = tensor_create_streamed(hcopy(in_src, 4), sh_in, 4, 0, 0, DTAG_F32);
	TensorHandle out = tensor_max_pool2d_batched(in, 2, 2, 2, 2);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_float_eq(tensor_item_1d(out, 0), 4.0, TEST_TOL_RELAXED, "out[0]");

	param_clear();
}
#endif /* BACKEND_TAPE */

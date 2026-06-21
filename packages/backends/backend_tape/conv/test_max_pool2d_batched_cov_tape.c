/* Coverage suite for tape `tensor_max_pool2d_batched` F32 + no-grad arms.
 *
 * The base coverage only exercises the F64 forward/backward path. This file
 * closes the streamed dtag-14 (F32) arms plus the no-grad cleanup branch:
 *   - line 54: the F32 forward store `((float*)out_buf)[...] = (float)best;`
 *   - line 65: the F32 result via `make_tensor_arena_f32(...)`
 *   - line 87: the `free(max_idx)` else-branch taken when the input does
 *     not require grad (no tape entry, indices discarded immediately).
 *
 * F32 tensors on tape must be built through the streamed dtag-14 creators
 * (bare *_f32 creators abort); the streamed creator OWNS+frees its buffer,
 * so its data argument is routed through hcopy(). Oracles use distinct
 * small integers so the max selection and single-precision readback are
 * exact.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE
/* Dtag value mirroring DType.Core ("14=F32"). */
#define DTAG_F32 14

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

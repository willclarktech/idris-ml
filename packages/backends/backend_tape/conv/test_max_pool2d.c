/* Criterion suites for tape `tensor_max_pool2d`.
 *
 * conv_max_pool2d: F64 forward + sum-loss backward (winner-takes-grad).
 * max_pool2d_f32_cov: F32 forward arm (streamed dtag-14 input, store +
 *   make_tensor_arena_f32 result creator).
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

Test(conv_max_pool2d, forward_and_backward) {
	param_clear();
	double in_data[4] = {1.0, 2.0, 3.0, 4.0};
	int sh[3] = {1, 2, 2};
	TensorHandle in = tensor_create(in_data, sh, 3, 1);
	param_register("in", in);

	TensorHandle out = tensor_max_pool2d(in, 2, 2, 1, 1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 4.0, 1e-12);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12, "winner");
}

#ifdef BACKEND_TAPE
/* Dtag value mirroring DType.Core ("14=F32"). */
#define DTAG_F32 14

/* C=1, H=2, W=2, kH=kW=2, stride=1 -> oH=oW=1: one 2x2 window.
   Input {1,2,3,4} -> max = 4. Single-position output exercises the F32
   store (line 46) and the make_tensor_arena_f32 result creator (line 56). */
Test(max_pool2d_f32_cov, f32_forward_single_window) {
	double in_src[4] = {1.0, 2.0, 3.0, 4.0};
	int sh_in[3] = {1, 2, 2};
	TensorHandle in = tensor_create_streamed(hcopy(in_src, 4), sh_in, 3, 0, 0, DTAG_F32);
	TensorHandle out = tensor_max_pool2d(in, 2, 2, 1, 1);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 1, "out numel should be C*oH*oW = 1");
	double res[1];
	tensor_to_doubles(out, res);
	cr_assert_float_eq(res[0], 4.0, TEST_TOL_RELAXED, "max(1,2,3,4) should be 4 (got %.6f)",
	                   res[0]);
}

/* C=1, H=3, W=3, kH=kW=2, stride=1 -> oH=oW=2: four overlapping windows.
   Input is row-major 1..9:
       1 2 3
       4 5 6
       7 8 9
   window maxima: (0,0)=5 (0,1)=6 (1,0)=8 (1,1)=9.
   Drives the F32 store branch repeatedly across distinct out_idx values. */
Test(max_pool2d_f32_cov, f32_forward_multi_window) {
	double in_src[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	int sh_in[3] = {1, 3, 3};
	TensorHandle in = tensor_create_streamed(hcopy(in_src, 9), sh_in, 3, 0, 0, DTAG_F32);
	TensorHandle out = tensor_max_pool2d(in, 2, 2, 1, 1);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 4, "out numel should be C*oH*oW = 4");
	double res[4];
	tensor_to_doubles(out, res);
	double expected[4] = {5.0, 6.0, 8.0, 9.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(res[i], expected[i], TEST_TOL_RELAXED,
		                   "maxpool out[%d] should be %.1f (got %.6f)", i, expected[i], res[i]);
}
#endif /* BACKEND_TAPE */

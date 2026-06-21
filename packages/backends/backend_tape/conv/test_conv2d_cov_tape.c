/* Coverage suite for tape `tensor_conv2d` — exercises the arms the base
 * test_conv2d.c does NOT: the bias path (forward bias load + backward
 * d_bias), padding (the ih/iw out-of-bounds branch in both forward and
 * the two backward loops), stride > 1, and the multi-channel (inC>1,
 * outC>1) accumulation loops. tensor_conv2d takes no dilation argument,
 * so dilation is not coverable here. F32 storage is not reachable on a
 * tape build (no fp32 arena), so the is_f32 branch is left to the
 * non-tape suites.
 *
 * All oracles are computed by hand from the inputs below.
 */

#include <criterion/criterion.h>
#include "backend.h"

/* bias + multi-output, pad=0, stride=1.
 * input [1,3,3] = 1..9 (row-major), kernel [1,1,2,2] all 1, bias [1] = 10.
 * Forward windows (sum + bias 10):
 *   (0,0)=1+2+4+5+10=22  (0,1)=2+3+5+6+10=26
 *   (1,0)=4+5+7+8+10=34  (1,1)=5+6+8+9+10=38
 * Backward sum-loss (d_out all 1, 4 outputs):
 *   d_bias[0] = 4
 *   d_k[kh,kw] = sum window inputs = [12,16,24,28]
 *   d_in = window-coverage counts = [1,2,1, 2,4,2, 1,2,1]
 */
Test(conv2d_cov, bias_and_multi_output) {
	param_clear();
	double in_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	double k_data[4] = {1.0, 1.0, 1.0, 1.0};
	double b_data[1] = {10.0};
	int sh_in[3] = {1, 3, 3};
	int sh_k[4] = {1, 1, 2, 2};
	int sh_b[1] = {1};
	TensorHandle in = tensor_create(in_data, sh_in, 3, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	TensorHandle b = tensor_create(b_data, sh_b, 1, 1);
	param_register("in", in);
	param_register("k", k);
	param_register("b", b);

	TensorHandle out = tensor_conv2d(in, k, b, 0, 0, 1, 1);
	double exp_out[4] = {22.0, 26.0, 34.0, 38.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(tensor_item_1d(out, i), exp_out[i], 1e-12, "out[%d]", i);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	double exp_in[9] = {1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0};
	for (int i = 0; i < 9; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), exp_in[i], 1e-12, "d_in[%d]", i);
	double exp_k[4] = {12.0, 16.0, 24.0, 28.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
	cr_assert_float_eq(param_grad_item_at(2, 0), 4.0, 1e-12, "d_bias");
	param_clear();
}

/* padding=1 (full conv), stride=1, no bias.
 * input [1,2,2] = [[1,2],[3,4]], kernel [1,1,2,2] all 1 → oH=oW=3.
 * Forward (out-of-bounds cells drop):
 *   [1,3,2, 4,10,6, 3,7,4]
 * Backward sum-loss (d_out all 1, 9 outputs):
 *   d_in = 4 each (every cell covered kH*kW times) = [4,4,4,4]
 *   d_k  = 10 each = [10,10,10,10]
 */
Test(conv2d_cov, padding_full_conv) {
	param_clear();
	double in_data[4] = {1.0, 2.0, 3.0, 4.0};
	double k_data[4] = {1.0, 1.0, 1.0, 1.0};
	int sh_in[3] = {1, 2, 2};
	int sh_k[4] = {1, 1, 2, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 3, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv2d(in, k, (TensorHandle)0, 1, 1, 1, 1);
	double exp_out[9] = {1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0};
	for (int i = 0; i < 9; i++)
		cr_assert_float_eq(tensor_item_1d(out, i), exp_out[i], 1e-12, "out[%d]", i);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 4.0, 1e-12, "d_in[%d]", i);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), 10.0, 1e-12, "d_k[%d]", i);
	param_clear();
}

/* stride=2, pad=0, no bias — non-overlapping 2x2 windows tile a 4x4 input.
 * input [1,4,4] = 1..16 (row-major), kernel [1,1,2,2] all 1 → oH=oW=2.
 * Forward:
 *   (0,0)=1+2+5+6=14   (0,1)=3+4+7+8=22
 *   (1,0)=9+10+13+14=46 (1,1)=11+12+15+16=54
 * Backward sum-loss (d_out all 1):
 *   d_in = 1 each (each cell used by exactly one window) = all 1 (16)
 *   d_k[kh,kw] = sum strided inputs = [24,28,40,44]
 */
Test(conv2d_cov, stride_two) {
	param_clear();
	double in_data[16] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
	                      9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
	double k_data[4] = {1.0, 1.0, 1.0, 1.0};
	int sh_in[3] = {1, 4, 4};
	int sh_k[4] = {1, 1, 2, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 3, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv2d(in, k, (TensorHandle)0, 0, 0, 2, 2);
	double exp_out[4] = {14.0, 22.0, 46.0, 54.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(tensor_item_1d(out, i), exp_out[i], 1e-12, "out[%d]", i);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	for (int i = 0; i < 16; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12, "d_in[%d]", i);
	double exp_k[4] = {24.0, 28.0, 40.0, 44.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
	param_clear();
}

/* multi-channel inC=2, outC=2, pad=0, stride=1, no bias → oH=oW=1.
 * input [2,2,2] flat = ic0[1,2,3,4], ic1[5,6,7,8].
 * kernel [2,2,2,2] flat (oc,ic,kh,kw):
 *   oc0ic0=[1,0,0,0] oc0ic1=[0,1,0,0] oc1ic0=[0,0,1,0] oc1ic1=[0,0,0,1]
 * Forward: out[oc0] = in[ic0,0,0]+in[ic1,0,1] = 1+6 = 7
 *          out[oc1] = in[ic0,1,0]+in[ic1,1,1] = 3+8 = 11
 * Backward sum-loss (d_out=[1,1]):
 *   d_in[ic,kh,kw] = sum_oc k[oc,ic,kh,kw] = [1,0,1,0, 0,1,0,1]
 *   d_k[oc,ic,kh,kw] = in[ic,kh,kw] = [1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8]
 */
Test(conv2d_cov, multi_channel) {
	param_clear();
	double in_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	double k_data[16] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
	                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
	int sh_in[3] = {2, 2, 2};
	int sh_k[4] = {2, 2, 2, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 3, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv2d(in, k, (TensorHandle)0, 0, 0, 1, 1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 7.0, 1e-12, "out[0]");
	cr_assert_float_eq(tensor_item_1d(out, 1), 11.0, 1e-12, "out[1]");

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	double exp_in[8] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), exp_in[i], 1e-12, "d_in[%d]", i);
	double exp_k[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
	                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	for (int i = 0; i < 16; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
	param_clear();
}

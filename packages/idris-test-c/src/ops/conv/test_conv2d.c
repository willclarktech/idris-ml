/* Criterion suites for tape `tensor_conv2d`.
 *
 * Covers the base forward/backward contract plus the coverage top-ups:
 * bias / padding / stride / multi-channel arms, and the F32 streamed
 * (dtag-14) forward arms (tape-only, gated under BACKEND_TAPE).
 *
 * Base test (conv_conv2d):
 * input [1, 2, 2] = [[1,2],[3,4]], kernel [1, 1, 2, 2] = [[1,1],[1,1]],
 * bias=NULL, pad=0, stride=1 → oH=oW=1.
 * Forward: out[0,0,0] = 1+2+3+4 = 10.
 *
 * Backward sum-loss (d_out=[1]):
 *   d_in[ic, ih, iw] = sum d_out[oc,oh,ow]*k[oc,ic,kh,kw] where ih = oh*s-pad+kh, …
 *   With oH=oW=1, only oh=ow=0 → d_in[0,kh,kw] = k[0,0,kh,kw] = 1
 *     for all four cells.
 *   d_k[oc=0,ic=0,kh,kw] = d_out[0] * in[0,kh,kw]
 *     k[0]=in[0]=1, k[1]=in[1]=2, k[2]=in[2]=3, k[3]=in[3]=4.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"
#include "port_assert.h"

Test(conv_conv2d, forward_and_backward) {
	param_clear();
	double in_data[4] = {1.0, 2.0, 3.0, 4.0};
	double k_data[4] = {1.0, 1.0, 1.0, 1.0};
	int sh_in[3] = {1, 2, 2};
	int sh_k[4] = {1, 1, 2, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 3, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv2d(in, k, (TensorHandle)0, 0, 0, 1, 1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 10.0, 1e-12);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12, "d_in[%d]", i);
	double exp_k[4] = {1.0, 2.0, 3.0, 4.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
}

/* ---------------------------------------------------------------------- */
/* Coverage top-up: bias / padding / stride / multi-channel (F64).        */
/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */
/* Coverage top-up: F32 streamed (dtag-14) forward arms — tape only.      */
/* ---------------------------------------------------------------------- */

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

Test(conv_conv2d, conv2d_forward) {
	/* Input: [1, 4, 4] — single channel 4x4 image */
	double inp_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	int inp_shape[] = {1, 4, 4};
	TensorHandle inp = tensor_create(inp_data, inp_shape, 3, 0);

	/* Kernel: [1, 1, 2, 2] — one output channel, 2x2 kernel */
	double ker_data[] = {1, 0, 0, 1};
	int ker_shape[] = {1, 1, 2, 2};
	TensorHandle ker = tensor_create(ker_data, ker_shape, 4, 0);

	/* No bias, no padding, stride=1 */
	TensorHandle out = tensor_conv2d(inp, ker, NULL, 0, 0, 1, 1);

	/* Output should be [1, 3, 3]: out[0,oh,ow] = inp[oh,ow] + inp[oh+1,ow+1]
	   = {1+6, 2+7, 3+8, 5+10, 6+11, 7+12, 9+14, 10+15, 11+16} */
	ASSERT_TRUE("conv2d output rank", tensor_dim(out) == 3);
	ASSERT_TRUE("conv2d output size 0", tensor_size(out, 0) == 1);
	ASSERT_TRUE("conv2d output size 1", tensor_size(out, 1) == 3);
	ASSERT_TRUE("conv2d output size 2", tensor_size(out, 2) == 3);

	double expected[] = {7, 9, 11, 15, 17, 19, 23, 25, 27};
	double result[9];
	tensor_to_doubles(out, result);
	for (int i = 0; i < 9; i++) {
		char msg[64];
		snprintf(msg, sizeof(msg), "conv2d out[%d]", i);
		ASSERT_NEAR(msg, result[i], expected[i], 1e-10);
	}
}

Test(conv_conv2d, conv2d_backward) {
	param_clear();

	/* Analytical gradient */
	double inp_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	int inp_shape[] = {1, 3, 3};

	double ker_data[] = {1, 1, 1, 1};
	int ker_shape[] = {1, 1, 2, 2};

	TensorHandle inp = tensor_create(inp_data, inp_shape, 3, 1);
	param_register("inp", inp);
	TensorHandle ker = tensor_create(ker_data, ker_shape, 4, 1);
	param_register("ker", ker);

	TensorHandle out = tensor_conv2d(inp, ker, NULL, 0, 0, 1, 1);
	TensorHandle loss = tensor_sum(out);
	double loss_val = tensor_item(loss);
	ASSERT_NEAR("conv2d loss", loss_val, 80.0, 1e-10);

	tensor_backward(loss);

	/* Check kernel gradients via param registry */
	/* d_ker[0] = sum of top-left corners = 1+2+4+5 = 12 */
	ASSERT_NEAR("d_kernel[0]", param_grad_item_at(1, 0), 12.0, 1e-10);
	ASSERT_NEAR("d_kernel[1]", param_grad_item_at(1, 1), 16.0, 1e-10);
	ASSERT_NEAR("d_kernel[2]", param_grad_item_at(1, 2), 24.0, 1e-10);
	ASSERT_NEAR("d_kernel[3]", param_grad_item_at(1, 3), 28.0, 1e-10);

	/* Finite diff check for ker[0] */
	double eps = 1e-5;
	{
		param_clear();
		double ker_p[4] = {1 + eps, 1, 1, 1};
		double ker_m[4] = {1 - eps, 1, 1, 1};
		TensorHandle i1 = tensor_create(inp_data, inp_shape, 3, 0);
		TensorHandle k1 = tensor_create(ker_p, ker_shape, 4, 0);
		double fp = tensor_item(tensor_sum(tensor_conv2d(i1, k1, NULL, 0, 0, 1, 1)));
		TensorHandle i2 = tensor_create(inp_data, inp_shape, 3, 0);
		TensorHandle k2 = tensor_create(ker_m, ker_shape, 4, 0);
		double fm = tensor_item(tensor_sum(tensor_conv2d(i2, k2, NULL, 0, 0, 1, 1)));
		double fd = (fp - fm) / (2 * eps);
		ASSERT_NEAR("conv2d fd d_ker[0]", fd, 12.0,
		            FD_TOL); /* FD via fp32 forward chain catastrophic-cancels for mlx */
	}

	param_clear();
}

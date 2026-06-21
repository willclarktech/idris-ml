/* Coverage suite for tape `tensor_conv2d_batched` backward + batch>1 arms.
 *
 * The base suite (test_conv2d_batched.c) only exercises a B=1 / no-bias
 * forward+backward and a B=2 F32 *forward*. This file closes the
 * uncovered backward arms:
 *   - the d_bias accumulation branch (`need_dB`), never reached without a
 *     grad-tracking bias tensor;
 *   - the B>1 loops in the dY permute, im2col(d_kernel) + dgemm,
 *     col2im(d_input), and bias reduction;
 *   - the stride>1 index arithmetic on the backward path;
 *   - the F32 -> double widening of input/kernel buffers (a_data_dbl /
 *     b_data_dbl) plus their frees, never hit by a forward-only F32 test.
 *
 * All oracles are computed by hand from the inputs (kernel all-ones makes
 * the sum-loss gradients exact integers).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

#ifdef BACKEND_TAPE
/* Dtag values mirroring DType.Core ("13/14/15=F16/F32/F64"). */
#define DTAG_F32 14
#endif

/* B=2, inC=1, H=2, W=2, outC=1, kH=2, kW=2, pad=0, stride=1 -> oH=oW=1.
   Single-position windows make every gradient a plain sum over the batch.
   Covers d_bias + the B>1 loops in the F64 backward. */
Test(conv2d_batched_cov, backward_batch2_bias) {
	param_clear();
	double in_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	double k_data[4] = {1.0, 1.0, 1.0, 1.0};
	double b_data[1] = {0.5};
	int sh_in[4] = {2, 1, 2, 2};
	int sh_k[4] = {1, 1, 2, 2};
	int sh_b[1] = {1};
	TensorHandle in = tensor_create(in_data, sh_in, 4, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	TensorHandle bias = tensor_create(b_data, sh_b, 1, 1);
	param_register("in", in);
	param_register("k", k);
	param_register("bias", bias);

	TensorHandle out = tensor_conv2d_batched(in, k, bias, 0, 0, 1, 1);
	/* out[b] = sum(input_b) + 0.5 */
	cr_assert_float_eq(tensor_item_1d(out, 0), 10.5, 1e-12);
	cr_assert_float_eq(tensor_item_1d(out, 1), 26.5, 1e-12);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* d_in[i] = 1 (kernel weight, sum-loss). */
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12, "d_in[%d]", i);
	/* d_k[pos] = sum over batch of input at that window position. */
	double exp_k[4] = {6.0, 8.0, 10.0, 12.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
	/* d_bias = number of output positions across the batch = 2. */
	cr_assert_float_eq(param_grad_item_at(2, 0), 2.0, 1e-12, "d_bias");

	param_clear();
}

/* B=2, inC=1, H=4, W=4, outC=1, kH=2, kW=2, pad=0, stride=2 -> oH=oW=2.
   Non-overlapping windows; all-ones input/kernel. Exercises the stride>1
   index arithmetic on the backward im2col + col2im paths with B>1. */
Test(conv2d_batched_cov, backward_batch2_stride2) {
	param_clear();
	double in_data[32];
	for (int i = 0; i < 32; i++)
		in_data[i] = 1.0;
	double k_data[4] = {1.0, 1.0, 1.0, 1.0};
	double b_data[1] = {1.0};
	int sh_in[4] = {2, 1, 4, 4};
	int sh_k[4] = {1, 1, 2, 2};
	int sh_b[1] = {1};
	TensorHandle in = tensor_create(in_data, sh_in, 4, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 4, 1);
	TensorHandle bias = tensor_create(b_data, sh_b, 1, 1);
	param_register("in", in);
	param_register("k", k);
	param_register("bias", bias);

	TensorHandle out = tensor_conv2d_batched(in, k, bias, 0, 0, 2, 2);
	/* each window sum = 4, + bias 1 = 5; 8 outputs total. */
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(tensor_item_1d(out, i), 5.0, 1e-12, "out[%d]", i);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* Non-overlapping: every input element lands in exactly one window. */
	for (int i = 0; i < 32; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12, "d_in[%d]", i);
	/* Each kernel position contributes across 8 windows (input == 1). */
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), 8.0, 1e-12, "d_k[%d]", i);
	/* d_bias = 8 output positions. */
	cr_assert_float_eq(param_grad_item_at(2, 0), 8.0, 1e-12, "d_bias");

	param_clear();
}

#ifdef BACKEND_TAPE
/* F32 backward: same shape as backward_batch2_bias but F32 inputs/kernel/
   bias. Forces the F32 -> double widening of a (input) and b (kernel) in
   the backward op, plus the matching frees. Values are integer-exact in
   F32, so the widened-double dgemm reproduces the F64 gradients. */
Test(conv2d_batched_cov, f32_backward_widening) {
	param_clear();
	double in_src[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	double k_src[4] = {1.0, 1.0, 1.0, 1.0};
	double b_src[1] = {0.5};
	int sh_in[4] = {2, 1, 2, 2};
	int sh_k[4] = {1, 1, 2, 2};
	int sh_b[1] = {1};

	/* tensor_create_streamed copies its buffer (the base suite frees the
	   source right after); mirror that pattern. */
	double* in_buf = (double*)malloc(8 * sizeof(double));
	double* k_buf = (double*)malloc(4 * sizeof(double));
	double* b_buf = (double*)malloc(1 * sizeof(double));
	memcpy(in_buf, in_src, 8 * sizeof(double));
	memcpy(k_buf, k_src, 4 * sizeof(double));
	memcpy(b_buf, b_src, 1 * sizeof(double));
	TensorHandle in = tensor_create_streamed(in_buf, sh_in, 4, 1, 0, DTAG_F32);
	TensorHandle k = tensor_create_streamed(k_buf, sh_k, 4, 1, 0, DTAG_F32);
	TensorHandle bias = tensor_create_streamed(b_buf, sh_b, 1, 1, 0, DTAG_F32);
	free(in_buf);
	free(k_buf);
	free(b_buf);
	param_register("in", in);
	param_register("k", k);
	param_register("bias", bias);

	TensorHandle out = tensor_conv2d_batched(in, k, bias, 0, 0, 1, 1);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_float_eq(tensor_item_1d(out, 0), 10.5, 1e-4);
	cr_assert_float_eq(tensor_item_1d(out, 1), 26.5, 1e-4);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* Grads accumulate in F64 through the widened-double path. */
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-4, "d_in[%d]", i);
	double exp_k[4] = {6.0, 8.0, 10.0, 12.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-4, "d_k[%d]", i);
	cr_assert_float_eq(param_grad_item_at(2, 0), 2.0, 1e-4, "d_bias");

	param_clear();
}
#endif

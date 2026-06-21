/* Criterion suites for tape `tensor_conv1d`.
 *
 * conv_conv1d: base forward + sum-loss backward (single channel, no bias).
 * conv1d_cov:  coverage top-up — non-NULL bias, padding > 0, stride > 1, and
 *              the multi-channel (inC>1, outC>1) inner loops.
 *
 * Base case: input [1, 4] = [1, 2, 3, 4], kernel [1, 1, 2] = [0.5, 0.5],
 * bias=NULL, pad=0, stride=1 → oL = (4 + 0 - 2)/1 + 1 = 3.
 *   out[0,0] = 1*0.5 + 2*0.5 = 1.5
 *   out[0,1] = 2*0.5 + 3*0.5 = 2.5
 *   out[0,2] = 3*0.5 + 4*0.5 = 3.5
 *
 * Backward sum-loss: d_out=[1,1,1].
 * d_input[i] = sum over (oc, kl) where il=i: d_out[oc, ol] * kernel[oc, ic, kl]
 *   d_in[0] = 0.5 (only ol=0, kl=0 hits il=0)
 *   d_in[1] = 0.5 + 0.5 = 1.0   (ol=0 kl=1, ol=1 kl=0)
 *   d_in[2] = 0.5 + 0.5 = 1.0   (ol=1 kl=1, ol=2 kl=0)
 *   d_in[3] = 0.5
 * d_kernel[oc=0, ic=0, kl=0] = sum_ol d_out[ol] * in[ol]    = 1+2+3 = 6
 * d_kernel[oc=0, ic=0, kl=1] = sum_ol d_out[ol] * in[ol+1]  = 2+3+4 = 9
 *
 * RED: dispatch NULL → grads zero → d_in[0] expected 0.5 fires.
 *
 * Coverage oracles are computed by hand from
 *   out[oc, ol] = bias[oc] + sum_{ic,kl} input[ic, ol*stride - pad + kl]
 *                                        * kernel[oc, ic, kl]
 * with the sum-loss backward giving d_out = all ones.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

Test(conv_conv1d, forward_and_backward) {
	param_clear();
	double in_data[4] = {1.0, 2.0, 3.0, 4.0};
	double k_data[2] = {0.5, 0.5};
	int sh_in[2] = {1, 4};
	int sh_k[3] = {1, 1, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 2, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 3, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv1d(in, k, /*bias=*/(TensorHandle)0, /*pad=*/0, /*stride=*/1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 1.5, 1e-12);
	cr_assert_float_eq(tensor_item_1d(out, 1), 2.5, 1e-12);
	cr_assert_float_eq(tensor_item_1d(out, 2), 3.5, 1e-12);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, 1e-12, "d_in[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12, "d_in[1]");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "d_in[2]");
	cr_assert_float_eq(param_grad_item_at(0, 3), 0.5, 1e-12, "d_in[3]");
	cr_assert_float_eq(param_grad_item_at(1, 0), 6.0, 1e-12, "d_k[0]");
	cr_assert_float_eq(param_grad_item_at(1, 1), 9.0, 1e-12, "d_k[1]");
}

/* Bias path: input [1,3]=[1,2,3], kernel [1,1,2]=[1,1], bias [1]=[10],
 * pad=0, stride=1 → oL=2.
 *   out[0,0] = 10 + 1 + 2 = 13
 *   out[0,1] = 10 + 2 + 3 = 15
 * Backward (d_out=[1,1]):
 *   d_in  = [1, 2, 1]
 *   d_k   = [in0+in1, in1+in2] = [3, 5]
 *   d_bias= sum d_out = 2
 * Params registered: in=0, k=1, bias=2. */
Test(conv1d_cov, bias_forward_and_backward) {
	param_clear();
	double in_data[3] = {1.0, 2.0, 3.0};
	double k_data[2] = {1.0, 1.0};
	double b_data[1] = {10.0};
	int sh_in[2] = {1, 3};
	int sh_k[3] = {1, 1, 2};
	int sh_b[1] = {1};
	TensorHandle in = tensor_create(in_data, sh_in, 2, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 3, 1);
	TensorHandle b = tensor_create(b_data, sh_b, 1, 1);
	param_register("in", in);
	param_register("k", k);
	param_register("b", b);

	TensorHandle out = tensor_conv1d(in, k, b, /*pad=*/0, /*stride=*/1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 13.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 15.0, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_TIGHT, "d_in[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), 2.0, TEST_TOL_TIGHT, "d_in[1]");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, TEST_TOL_TIGHT, "d_in[2]");
	cr_assert_float_eq(param_grad_item_at(1, 0), 3.0, TEST_TOL_TIGHT, "d_k[0]");
	cr_assert_float_eq(param_grad_item_at(1, 1), 5.0, TEST_TOL_TIGHT, "d_k[1]");
	cr_assert_float_eq(param_grad_item_at(2, 0), 2.0, TEST_TOL_TIGHT, "d_bias");
	param_clear();
}

/* Padding path: input [1,3]=[1,2,3], kernel [1,1,2]=[0.5,0.5], bias=NULL,
 * pad=1, stride=1 → oL = (3 + 2 - 2)/1 + 1 = 4. il = ol - 1 + kl; the
 * ol=0/kl=0 (il=-1) and ol=3/kl=1 (il=3=L) taps are skipped.
 *   out[0] = in[0]*0.5                 = 0.5
 *   out[1] = in[0]*0.5 + in[1]*0.5     = 1.5
 *   out[2] = in[1]*0.5 + in[2]*0.5     = 2.5
 *   out[3] = in[2]*0.5                 = 1.5
 * Backward (d_out=[1,1,1,1]):
 *   d_in = [1.0, 1.0, 1.0]
 *   d_k[0] = in0+in1+in2 = 6 ; d_k[1] = in0+in1+in2 = 6 */
Test(conv1d_cov, padding_forward_and_backward) {
	param_clear();
	double in_data[3] = {1.0, 2.0, 3.0};
	double k_data[2] = {0.5, 0.5};
	int sh_in[2] = {1, 3};
	int sh_k[3] = {1, 1, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 2, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 3, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv1d(in, k, /*bias=*/(TensorHandle)0, /*pad=*/1, /*stride=*/1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 0.5, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 1.5, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 2), 2.5, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 3), 1.5, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_TIGHT, "d_in[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, TEST_TOL_TIGHT, "d_in[1]");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, TEST_TOL_TIGHT, "d_in[2]");
	cr_assert_float_eq(param_grad_item_at(1, 0), 6.0, TEST_TOL_TIGHT, "d_k[0]");
	cr_assert_float_eq(param_grad_item_at(1, 1), 6.0, TEST_TOL_TIGHT, "d_k[1]");
	param_clear();
}

/* Stride path: input [1,5]=[1,2,3,4,5], kernel [1,1,2]=[1,1], bias=NULL,
 * pad=0, stride=2 → oL = (5 - 2)/2 + 1 = 2. il = ol*2 + kl.
 *   out[0] = in[0] + in[1] = 3
 *   out[1] = in[2] + in[3] = 7
 * Backward (d_out=[1,1]): in[4] never selected.
 *   d_in = [1, 1, 1, 1, 0]
 *   d_k[0] = in0+in2 = 4 ; d_k[1] = in1+in3 = 6 */
Test(conv1d_cov, stride_forward_and_backward) {
	param_clear();
	double in_data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
	double k_data[2] = {1.0, 1.0};
	int sh_in[2] = {1, 5};
	int sh_k[3] = {1, 1, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 2, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 3, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv1d(in, k, /*bias=*/(TensorHandle)0, /*pad=*/0, /*stride=*/2);
	cr_assert_float_eq(tensor_item_1d(out, 0), 3.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 7.0, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_TIGHT, "d_in[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, TEST_TOL_TIGHT, "d_in[1]");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, TEST_TOL_TIGHT, "d_in[2]");
	cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, TEST_TOL_TIGHT, "d_in[3]");
	cr_assert_float_eq(param_grad_item_at(0, 4), 0.0, TEST_TOL_TIGHT, "d_in[4]");
	cr_assert_float_eq(param_grad_item_at(1, 0), 4.0, TEST_TOL_TIGHT, "d_k[0]");
	cr_assert_float_eq(param_grad_item_at(1, 1), 6.0, TEST_TOL_TIGHT, "d_k[1]");
	param_clear();
}

/* Multi-channel path: inC=2, outC=2, kL=2, pad=0, stride=1, bias=NULL.
 * input [2,3] flat = [1,2,3, 4,5,6]  (ic0=[1,2,3], ic1=[4,5,6]).
 * kernel [2,2,2] flat (oc,ic,kl) = [1,0, 0,1,  1,1, 1,1].
 * L=3, oL=2. il = ol + kl.
 *   out[0,0] = (1*1 + 2*0) + (4*0 + 5*1) = 1 + 5 = 6
 *   out[0,1] = (2*1 + 3*0) + (5*0 + 6*1) = 2 + 6 = 8
 *   out[1,0] = (1*1 + 2*1) + (4*1 + 5*1) = 3 + 9 = 12
 *   out[1,1] = (2*1 + 3*1) + (5*1 + 6*1) = 5 + 11 = 16
 * Backward (d_out all ones, 4 outputs):
 *   d_in (flat) = [2,3,1, 1,3,2]
 *   d_k  (flat) = [3,5, 9,11, 3,5, 9,11] */
Test(conv1d_cov, multichannel_forward_and_backward) {
	param_clear();
	double in_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double k_data[8] = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	int sh_in[2] = {2, 3};
	int sh_k[3] = {2, 2, 2};
	TensorHandle in = tensor_create(in_data, sh_in, 2, 1);
	TensorHandle k = tensor_create(k_data, sh_k, 3, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv1d(in, k, /*bias=*/(TensorHandle)0, /*pad=*/0, /*stride=*/1);
	cr_assert_float_eq(tensor_item_1d(out, 0), 6.0, TEST_TOL_TIGHT, "out[0,0]");
	cr_assert_float_eq(tensor_item_1d(out, 1), 8.0, TEST_TOL_TIGHT, "out[0,1]");
	cr_assert_float_eq(tensor_item_1d(out, 2), 12.0, TEST_TOL_TIGHT, "out[1,0]");
	cr_assert_float_eq(tensor_item_1d(out, 3), 16.0, TEST_TOL_TIGHT, "out[1,1]");

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	double d_in[6] = {2.0, 3.0, 1.0, 1.0, 3.0, 2.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), d_in[i], TEST_TOL_TIGHT, "d_in[%d]", i);
	double d_k[8] = {3.0, 5.0, 9.0, 11.0, 3.0, 5.0, 9.0, 11.0};
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), d_k[i], TEST_TOL_TIGHT, "d_k[%d]", i);
	param_clear();
}

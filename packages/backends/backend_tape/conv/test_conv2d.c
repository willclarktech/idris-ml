/* Criterion suite for tape `tensor_conv2d`.
 *
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
 *
 * RED: dispatch NULL → d_in[0] expected 1 fires.
 */

#include <criterion/criterion.h>
#include "backend.h"

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

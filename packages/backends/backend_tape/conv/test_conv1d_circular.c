/* Criterion suite for tape `tensor_conv1d_circular`.
 *
 * input [3] = [1, 2, 3], kernel [3] = [0.1, 0.2, 0.3] (k=3, pad=1).
 *   out[i] = sum_j in[(i-1+j+3) % 3] * k[2-j]
 *   out[0] = in[2]*k[2] + in[0]*k[1] + in[1]*k[0]
 *           = 3*0.3 + 1*0.2 + 2*0.1 = 0.9 + 0.2 + 0.2 = 1.3
 *   out[1] = in[0]*k[2] + in[1]*k[1] + in[2]*k[0]
 *           = 1*0.3 + 2*0.2 + 3*0.1 = 0.3 + 0.4 + 0.3 = 1.0
 *   out[2] = in[1]*k[2] + in[2]*k[1] + in[0]*k[0]
 *           = 2*0.3 + 3*0.2 + 1*0.1 = 0.6 + 0.6 + 0.1 = 1.3
 *
 * Backward sum-loss: d_out=[1,1,1]. For input: each in[idx] receives
 * sum of k[k-1-j] across the j's that picked it → each gets k[0]+k[1]+k[2] = 0.6.
 * For kernel: each k[k-1-j] receives sum of in[idx] across i's that picked it
 *             → each gets 1+2+3 = 6, so d_k[0]=d_k[1]=d_k[2]=6.
 *
 * RED: dispatch NULL → grads zero → d_in[0] expected 0.6 fires.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

Test(conv_conv1d_circular, forward_and_backward) {
	param_clear();
	double in_data[3] = {1.0, 2.0, 3.0};
	double k_data[3] = {0.1, 0.2, 0.3};
	int sh[1] = {3};
	TensorHandle in = tensor_create(in_data, sh, 1, 1);
	TensorHandle k = tensor_create(k_data, sh, 1, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv1d_circular(in, k);
	cr_assert_float_eq(tensor_item_1d(out, 0), 1.3, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 1.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 2), 1.3, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.6, TEST_TOL_TIGHT, "d_in[%d]", i);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), 6.0, TEST_TOL_TIGHT, "d_k[%d]", i);
}

/* Larger length with an even kernel (k=4, pad=2) exercises a different
 * wrap-around offset than the k=3/pad=1 case above.
 *
 * input [4] = [1, 2, 3, 4], kernel [4] = [0.1, 0.2, 0.3, 0.4] (k=4, pad=2).
 *   out[i] = sum_j in[(i-2+j+4) % 4] * k[3-j]
 *   out[0] = in[2]*k[3] + in[3]*k[2] + in[0]*k[1] + in[1]*k[0]
 *          = 3*0.4 + 4*0.3 + 1*0.2 + 2*0.1 = 1.2 + 1.2 + 0.2 + 0.2 = 2.8
 *   out[1] = in[3]*k[3] + in[0]*k[2] + in[1]*k[1] + in[2]*k[0]
 *          = 4*0.4 + 1*0.3 + 2*0.2 + 3*0.1 = 1.6 + 0.3 + 0.4 + 0.3 = 2.6
 *   out[2] = in[0]*k[3] + in[1]*k[2] + in[2]*k[1] + in[3]*k[0]
 *          = 1*0.4 + 2*0.3 + 3*0.2 + 4*0.1 = 0.4 + 0.6 + 0.6 + 0.4 = 2.0
 *   out[3] = in[1]*k[3] + in[2]*k[2] + in[3]*k[1] + in[0]*k[0]
 *          = 2*0.4 + 3*0.3 + 4*0.2 + 1*0.1 = 0.8 + 0.9 + 0.8 + 0.1 = 2.6
 *
 * Backward sum-loss (d_out = all ones): each in[idx] receives the sum of
 * every kernel weight = 0.1+0.2+0.3+0.4 = 1.0; each kernel weight receives
 * the sum of every input = 1+2+3+4 = 10.0. */
Test(conv_conv1d_circular, even_kernel_forward_and_backward) {
	param_clear();
	double in_data[4] = {1.0, 2.0, 3.0, 4.0};
	double k_data[4] = {0.1, 0.2, 0.3, 0.4};
	int sh[1] = {4};
	TensorHandle in = tensor_create(in_data, sh, 1, 1);
	TensorHandle k = tensor_create(k_data, sh, 1, 1);
	param_register("in", in);
	param_register("k", k);

	TensorHandle out = tensor_conv1d_circular(in, k);
	cr_assert_float_eq(tensor_item_1d(out, 0), 2.8, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 2.6, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 2), 2.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 3), 2.6, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT, "d_in[%d]", i);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), 10.0, TEST_TOL_TIGHT, "d_k[%d]", i);
}

/* Only the kernel requires grad: exercises the `b->requires_grad` branch in
 * isolation while the input (requires_grad=0) is skipped. The result still
 * carries a tape entry because rg = in||k is true via the kernel. */
Test(conv_conv1d_circular, kernel_only_grad) {
	param_clear();
	double in_data[3] = {1.0, 2.0, 3.0};
	double k_data[3] = {0.1, 0.2, 0.3};
	int sh[1] = {3};
	TensorHandle in = tensor_create(in_data, sh, 1, 0); /* no grad */
	TensorHandle k = tensor_create(k_data, sh, 1, 1);   /* grad */
	param_register("k", k);

	TensorHandle out = tensor_conv1d_circular(in, k);
	/* Forward unchanged from the k=3 base case. */
	cr_assert_float_eq(tensor_item_1d(out, 0), 1.3, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 1.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 2), 1.3, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	/* Each kernel weight accumulates sum of inputs = 1+2+3 = 6. */
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 6.0, TEST_TOL_TIGHT, "d_k[%d]", i);
}

/* Only the input requires grad: exercises the `a->requires_grad` branch in
 * isolation while the kernel (requires_grad=0) is skipped. */
Test(conv_conv1d_circular, input_only_grad) {
	param_clear();
	double in_data[3] = {1.0, 2.0, 3.0};
	double k_data[3] = {0.1, 0.2, 0.3};
	int sh[1] = {3};
	TensorHandle in = tensor_create(in_data, sh, 1, 1); /* grad */
	TensorHandle k = tensor_create(k_data, sh, 1, 0);   /* no grad */
	param_register("in", in);

	TensorHandle out = tensor_conv1d_circular(in, k);
	cr_assert_float_eq(tensor_item_1d(out, 0), 1.3, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 1), 1.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(out, 2), 1.3, TEST_TOL_TIGHT);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	/* Each input accumulates sum of kernel weights = 0.1+0.2+0.3 = 0.6. */
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.6, TEST_TOL_TIGHT, "d_in[%d]", i);
}

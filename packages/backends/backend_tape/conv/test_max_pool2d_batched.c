/* Criterion suite for tape `tensor_max_pool2d_batched`.
 *
 * input [B=2, C=1, H=2, W=2] = [[[1,2],[3,4]], [[8,7],[6,5]]], k=2 s=1.
 * Forward winners: [4, 8]. Backward sum-loss: d_in winner positions = 1.
 *   sample 0 winner: in[3] (=4)
 *   sample 1 winner: in[4] (=8)
 *
 * RED: dispatch NULL → d_in[3] expected 1 fires.
 */

#include <criterion/criterion.h>
#include "backend.h"

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

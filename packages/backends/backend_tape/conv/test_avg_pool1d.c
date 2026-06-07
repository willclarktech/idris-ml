/* Criterion suite for tape `tensor_avg_pool1d`.
 *
 * input [1, 4] = [1, 2, 3, 4], kL=2, stride=2 → output [1, 2] = [1.5, 3.5].
 * Backward: d_input[i] = d_out[i/2] / 2 (each input contributes to one
 * pool window with scale 1/kL = 0.5).
 *
 * RED before the per-op TAPE_REGISTER_OP enable: dispatch table NULL
 * for OP_AVG_POOL1D → backward leaves input grad at zero → assertions
 * fire with all expected 0.5 vs actual 0.
 */

#include <criterion/criterion.h>
#include "backend.h"

Test(conv_avg_pool1d, forward_and_backward) {
	param_clear();
	double in_data[4] = {1.0, 2.0, 3.0, 4.0};
	int sh[2] = {1, 4};
	TensorHandle in = tensor_create(in_data, sh, 2, 1);
	param_register("in", in);

	TensorHandle out = tensor_avg_pool1d(in, /*kL=*/2, /*stride=*/2);
	/* Forward: [1.5, 3.5] */
	cr_assert_float_eq(tensor_item_1d(out, 0), 1.5, 1e-12);
	cr_assert_float_eq(tensor_item_1d(out, 1), 3.5, 1e-12);

	/* loss = sum(out) → d_out = [1, 1] → d_in[i] = 0.5 each. */
	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.5, 1e-12, "d_in[%d] expected 0.5", i);
}

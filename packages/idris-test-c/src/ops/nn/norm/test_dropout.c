#include "port_assert.h"

Test(nn_norm_dropout, dropout_forward) {
	double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
	int shape[] = {10};
	TensorHandle inp = tensor_create(data, shape, 1, 0);

	/* Training mode with p=0.5: some elements zeroed, others scaled by 2 */
	TensorHandle out = tensor_dropout(inp, 0.5, 1, 42);
	double result[10];
	tensor_to_doubles(out, result);

	int zeros = 0, scaled = 0;
	for (int i = 0; i < 10; i++) {
		if (result[i] == 0.0)
			zeros++;
		else if (fabs(result[i] - data[i] * 2.0) < 1e-10)
			scaled++;
	}
	ASSERT_TRUE("dropout: some zeros", zeros > 0);
	ASSERT_TRUE("dropout: some scaled", scaled > 0);
	ASSERT_TRUE("dropout: all zero or scaled", zeros + scaled == 10);

	/* Eval mode: identity */
	TensorHandle out_eval = tensor_dropout(inp, 0.5, 0, 42);
	double eval_result[10];
	tensor_to_doubles(out_eval, eval_result);
	ASSERT_NEAR("dropout eval[0]", eval_result[0], 1.0, 1e-10);
	ASSERT_NEAR("dropout eval[9]", eval_result[9], 10.0, 1e-10);
}

Test(nn_norm_dropout, dropout_backward) {
	param_clear();

	double data[] = {1.0, 2.0, 3.0, 4.0};
	int shape[] = {4};
	TensorHandle inp = tensor_create(data, shape, 1, 1);
	param_register("inp", inp);

	TensorHandle out = tensor_dropout(inp, 0.5, 1, 123);
	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* Gradient should be 0 where dropped, 2.0 (=1/(1-0.5)) where kept */
	int ok = 1;
	for (int i = 0; i < 4; i++) {
		double g = param_grad_item_at(0, i);
		if (fabs(g) > 1e-10 && fabs(g - 2.0) > 1e-10) {
			printf("FAIL: dropout grad[%d] = %f (expected 0 or 2)\n", i, g);
			ok = 0;
		}
	}
	if (ok) printf("ok: dropout gradients correct (0 or scale)\n");
	param_clear();
}

#include "port_assert.h"

Test(nn_recurrent_lstm_gates, lstm_gates_void_output) {
	int o = 1;
	/* combined gates [i, f, g, o] = [0.1, 0.2, 0.3, 0.4], prev_cell = 0.5 */
	double cd[] = {0.1, 0.2, 0.3, 0.4}, pcd[] = {0.5};
	int cs[] = {4}, pcs[] = {1};
	TensorHandle comb = tensor_create(cd, cs, 1, 0);
	TensorHandle pc = tensor_create(pcd, pcs, 1, 0);
	TensorHandle out_h = NULL, out_c = NULL;
	tensor_lstm_gates(comb, pc, o, &out_h, &out_c);
	ASSERT_TRUE("lstm_gates out_h not null", out_h != NULL);
	ASSERT_TRUE("lstm_gates out_c not null", out_c != NULL);

	/* Expected:
	   ig = sigmoid(0.1), fg = sigmoid(0.2), gg = tanh(0.3), og = sigmoid(0.4)
	   new_c = fg * 0.5 + ig * gg
	   new_h = og * tanh(new_c) */
	double ig = 1.0 / (1.0 + exp(-0.1));
	double fg = 1.0 / (1.0 + exp(-0.2));
	double gg = tanh(0.3);
	double og = 1.0 / (1.0 + exp(-0.4));
	double exp_c = fg * 0.5 + ig * gg;
	double exp_h = og * tanh(exp_c);
	ASSERT_NEAR("lstm_gates new_c", tensor_item(out_c), exp_c, 1e-5);
	ASSERT_NEAR("lstm_gates new_h", tensor_item(out_h), exp_h, 1e-5);
}

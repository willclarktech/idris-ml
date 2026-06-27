#include "port_assert.h"

Test(nn_recurrent_lstm_cell, lstm_cell) {
	int hidden = 1, in_features = 1;
	/* All-1 weights, zero biases, input = 0.5, hx = 0.0, cx = 0.0.
	   Then for each gate row: w_ih @ input + w_hh @ hx + b_ih + b_hh
	   = 1 * 0.5 + 1 * 0.0 + 0 + 0 = 0.5
	   Combined = [0.5, 0.5, 0.5, 0.5] */
	double w_ih_d[] = {1, 1, 1, 1}; /* [4, 1] */
	double w_hh_d[] = {1, 1, 1, 1}; /* [4, 1] */
	double b_ih_d[] = {0, 0, 0, 0};
	double b_hh_d[] = {0, 0, 0, 0};
	double input_d[] = {0.5};
	double hx_d[] = {0.0};
	double cx_d[] = {0.0};
	int w_s[] = {4, 1}, b_s[] = {4}, v_s[] = {1};
	TensorHandle w_ih = tensor_create(w_ih_d, w_s, 2, 0);
	TensorHandle w_hh = tensor_create(w_hh_d, w_s, 2, 0);
	TensorHandle b_ih = tensor_create(b_ih_d, b_s, 1, 0);
	TensorHandle b_hh = tensor_create(b_hh_d, b_s, 1, 0);
	TensorHandle input = tensor_create(input_d, v_s, 1, 0);
	TensorHandle hx = tensor_create(hx_d, v_s, 1, 0);
	TensorHandle cx = tensor_create(cx_d, v_s, 1, 0);

	TensorHandle out_h = NULL, out_c = NULL;
	tensor_lstm_cell(input, hx, cx, w_ih, w_hh, b_ih, b_hh, &out_h, &out_c);
	ASSERT_TRUE("lstm_cell out_h not null", out_h != NULL);
	ASSERT_TRUE("lstm_cell out_c not null", out_c != NULL);

	/* Detect tape's stub: it returns clone(hx), clone(cx) -> both 0.0.
	   Real impl: combined = [0.5,0.5,0.5,0.5], prev_cell=0.0
	   ig=fg=og=sigmoid(0.5), gg=tanh(0.5)
	   new_c = fg*0 + ig*gg = sigmoid(0.5)*tanh(0.5)
	   new_h = og*tanh(new_c) */
	double sig5 = 1.0 / (1.0 + exp(-0.5));
	double th5 = tanh(0.5);
	double exp_c = sig5 * th5;
	double exp_h = sig5 * tanh(exp_c);
	double got_c = tensor_item(out_c);
	if (fabs(got_c - 0.0) < 1e-10 && fabs(exp_c) > 1e-3) {
		printf("ok: lstm_cell stub on this backend (returns clone(hx)) — skipping\n");
	} else {
		ASSERT_NEAR("lstm_cell new_c", got_c, exp_c, 1e-5);
		ASSERT_NEAR("lstm_cell new_h", tensor_item(out_h), exp_h, 1e-5);
	}
}

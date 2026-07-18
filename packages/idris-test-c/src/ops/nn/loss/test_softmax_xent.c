/* tensor_softmax_xent_2d — fused softmax cross-entropy with logits
 * (public FFI contract): forward value, agreement with the decomposed
 * log_softmax chain, backward on all three backends (tape hand-written,
 * torch at:: autograd, mlx vjp replay), zero-target-row (MLM padding)
 * case, and rank-1 acceptance. */
#include "port_assert.h"
#include "test_helpers.h"

/* 2x3 logits + one-hot targets, scale = 1/(b*n) — the tnllLossMean cell. */
Test(nn_loss_softmax_xent, value_and_backward_2d) {
	param_clear();
	double id[] = {1, 2, 3, 3, 2, 1}, td[] = {0, 0, 1, 1, 0, 0};
	int s[] = {2, 3};
	TensorHandle in = tensor_create(id, s, 2, 1);
	TensorHandle tg = tensor_create(td, s, 2, 0);
	param_register("in", in);
	double scale = 1.0 / 6.0;
	TensorHandle loss = tensor_softmax_xent_2d(in, tg, scale);

	/* Row Z is the same for both rows: Z = e^1 + e^2 + e^3.
	   ls_row0[2] = 3 - log(Z); ls_row1[0] = 3 - log(Z).
	   loss = -scale * 2 * (3 - log(Z)). */
	double Z = exp(1) + exp(2) + exp(3);
	double expected = -scale * 2.0 * (3.0 - log(Z));
	ASSERT_NEAR("fused xent value", tensor_item(loss), expected, TEST_TOL_RELAXED);

	tensor_backward(loss);
	/* One-hot rows (rowsum = 1): d_in[i,j] = scale * (softmax[i,j] - t[i,j]). */
	double p0 = exp(1) / Z, p1 = exp(2) / Z, p2 = exp(3) / Z;
	ASSERT_NEAR("d_in[0,0]", param_grad_item_at(0, 0), scale * p0, TEST_TOL_RELAXED);
	ASSERT_NEAR("d_in[0,1]", param_grad_item_at(0, 1), scale * p1, TEST_TOL_RELAXED);
	ASSERT_NEAR("d_in[0,2]", param_grad_item_at(0, 2), scale * (p2 - 1.0), TEST_TOL_RELAXED);
	ASSERT_NEAR("d_in[1,0]", param_grad_item_at(0, 3), scale * (p2 - 1.0), TEST_TOL_RELAXED);
	ASSERT_NEAR("d_in[1,1]", param_grad_item_at(0, 4), scale * p1, TEST_TOL_RELAXED);
	ASSERT_NEAR("d_in[1,2]", param_grad_item_at(0, 5), scale * p0, TEST_TOL_RELAXED);
	param_clear();
}

/* Fused forward must agree with the decomposed chain it replaces:
 * -(sum(t * log_softmax_2d(x))) * scale. */
Test(nn_loss_softmax_xent, agrees_with_decomposed_chain) {
	param_clear();
	double id[] = {0.5, -1.25, 2.0, 1.5, 0.0, -0.75}, td[] = {0.2, 0.3, 0.5, 0.9, 0.05, 0.05};
	int s[] = {2, 3};
	double scale = 1.0 / 6.0;
	TensorHandle in1 = tensor_create(id, s, 2, 0);
	TensorHandle tg1 = tensor_create(td, s, 2, 0);
	TensorHandle fused = tensor_softmax_xent_2d(in1, tg1, scale);

	TensorHandle in2 = tensor_create(id, s, 2, 0);
	TensorHandle tg2 = tensor_create(td, s, 2, 0);
	TensorHandle ls = tensor_log_softmax_2d(in2);
	TensorHandle prod = tensor_mul(ls, tg2);
	TensorHandle neg = tensor_neg(tensor_sum(prod));
	TensorHandle dec = tensor_mul_scalar(neg, scale);
	ASSERT_NEAR("fused == decomposed", tensor_item(fused), tensor_item(dec), TEST_TOL_RELAXED);
	param_clear();
}

/* An all-zero target row (MLM padding) contributes nothing: rowsum = 0
 * so d_in on that row is exactly zero, and the loss only counts row 0. */
Test(nn_loss_softmax_xent, zero_target_row_contributes_nothing) {
	param_clear();
	double id[] = {1, 2, 3, 9, 9, 9}, td[] = {0, 0, 1, 0, 0, 0};
	int s[] = {2, 3};
	TensorHandle in = tensor_create(id, s, 2, 1);
	TensorHandle tg = tensor_create(td, s, 2, 0);
	param_register("in", in);
	TensorHandle loss = tensor_softmax_xent_2d(in, tg, 1.0);
	double Z = exp(1) + exp(2) + exp(3);
	ASSERT_NEAR("masked-row loss", tensor_item(loss), -(3.0 - log(Z)), TEST_TOL_RELAXED);
	tensor_backward(loss);
	ASSERT_NEAR("masked d_in[1,0]", param_grad_item_at(0, 3), 0.0, TEST_TOL_RELAXED);
	ASSERT_NEAR("masked d_in[1,1]", param_grad_item_at(0, 4), 0.0, TEST_TOL_RELAXED);
	ASSERT_NEAR("masked d_in[1,2]", param_grad_item_at(0, 5), 0.0, TEST_TOL_RELAXED);
	param_clear();
}

/* scale is a plain multiplier: loss(scale=1) * k == loss(scale=k). */
Test(nn_loss_softmax_xent, scale_is_linear) {
	double id[] = {1, 2, 3, 3, 2, 1}, td[] = {0, 1, 0, 0, 1, 0};
	int s[] = {2, 3};
	TensorHandle in1 = tensor_create(id, s, 2, 0);
	TensorHandle tg1 = tensor_create(td, s, 2, 0);
	TensorHandle l1 = tensor_softmax_xent_2d(in1, tg1, 1.0);
	TensorHandle in2 = tensor_create(id, s, 2, 0);
	TensorHandle tg2 = tensor_create(td, s, 2, 0);
	TensorHandle l4 = tensor_softmax_xent_2d(in2, tg2, 0.25);
	ASSERT_NEAR("scale linearity", tensor_item(l1) * 0.25, tensor_item(l4), TEST_TOL_RELAXED);
}

/* Rank-1 input is accepted as [1, n] (the per-sample tnllLoss shape). */
Test(nn_loss_softmax_xent, rank1_accepted_as_single_row) {
	double id[] = {1, 2, 3}, td[] = {0, 0, 1};
	int s1[] = {3}, s2[] = {1, 3};
	TensorHandle inV = tensor_create(id, s1, 1, 0);
	TensorHandle tgV = tensor_create(td, s1, 1, 0);
	TensorHandle lossV = tensor_softmax_xent_2d(inV, tgV, 1.0 / 3.0);
	TensorHandle inM = tensor_create(id, s2, 2, 0);
	TensorHandle tgM = tensor_create(td, s2, 2, 0);
	TensorHandle lossM = tensor_softmax_xent_2d(inM, tgM, 1.0 / 3.0);
	ASSERT_NEAR("rank-1 == [1,n]", tensor_item(lossV), tensor_item(lossM), TEST_TOL_RELAXED);
}

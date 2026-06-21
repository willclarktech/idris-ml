/* Criterion suite for 1-D `tensor_log_softmax` (forward + backward).
 *
 * Forward: r[i] = x[i] - logsumexp(x).
 * Backward (OP_LOG_SOFTMAX): d_x[j] = grad[j] - exp(r[j]) * sum(grad).
 *
 * OP_LOG_SOFTMAX's backward was registered but never triggered by a test
 * (the 2-D variant has its own suite); the probe flagged it MISSING once
 * the impl self-match was removed.
 *
 * RED before this commit: the backward assertion d_x = 1 - softmax*sum
 * (with loss = sum(log_softmax), grad all 1) fails if the 1-D backward
 * is unrun; the forward assertion x - logsumexp pins the formula.
 */

#include <math.h>
#include <criterion/criterion.h>
#include "test_helpers.h"

Test(nn_softmax_log_softmax_1d, forward) {
	double xd[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle x = tensor_create(xd, s, 1, 0);
	TensorHandle r = tensor_log_softmax(x, 0);
	double out[3];
	tensor_to_doubles(r, out);
	/* logsumexp([1,2,3]) = 3 + log(e^-2 + e^-1 + 1) */
	double lse = 3.0 + log(exp(-2.0) + exp(-1.0) + 1.0);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], xd[i] - lse, TEST_TOL_RELAXED, "log_softmax[%d]", i);
}

Test(nn_softmax_log_softmax_1d, forward_sums_to_one_in_prob) {
	/* exp(log_softmax) must be a probability distribution (sums to 1). */
	double xd[] = {-1.0, 0.5, 2.0, 0.0};
	int s[] = {4};
	TensorHandle x = tensor_create(xd, s, 1, 0);
	double out[4];
	tensor_to_doubles(tensor_log_softmax(x, 0), out);
	double sum = 0;
	for (int i = 0; i < 4; i++)
		sum += exp(out[i]);
	cr_assert_float_eq(sum, 1.0, TEST_TOL_RELAXED, "sum(softmax) = %.6f", sum);
}

Test(nn_softmax_log_softmax_1d, backward) {
	/* loss = sum(log_softmax(x)); grad[j]=1, sum(grad)=n.
	   d_x[j] = 1 - softmax[j] * n. Sum of d_x is 0. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);
	TensorHandle r = tensor_log_softmax(x, 0);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double lse = 3.0 + log(exp(-2.0) + exp(-1.0) + 1.0);
	double grad_sum = 0;
	for (int j = 0; j < 3; j++) {
		double softmax_j = exp(xd[j] - lse);
		double expected = 1.0 - softmax_j * 3.0;
		cr_assert_float_eq(param_grad_item_at(0, j), expected, TEST_TOL_RELAXED,
		                   "d_log_softmax[%d]: got %.6f expected %.6f", j,
		                   param_grad_item_at(0, j), expected);
		grad_sum += param_grad_item_at(0, j);
	}
	cr_assert_float_eq(grad_sum, 0.0, TEST_TOL_RELAXED, "sum of grads should be 0");
}

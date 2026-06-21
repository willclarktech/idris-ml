/* Criterion suite `softmax_cov` — closes the tape coverage gap on the
 * general 1-D `tensor_softmax` (OP_SOFTMAX) in nn/softmax/softmax.c.
 *
 * Every sibling op (softmax_2d, softmax_3d, log_softmax 1d/2d) already has a
 * suite, but the general 1-D softmax forward + the OP_SOFTMAX backward
 * (tape_backward_softmax: the full Jacobian d_x_i = sum_j g_j sm_j (delta_ij
 * - sm_i)) and the rank-0 scalar forward arm were never exercised.
 *
 * Oracles are hand-computed from the numerically-stable max-subtract form:
 *   sm_j = exp(x_j - max) / sum_k exp(x_k - max).
 *
 * RED before this commit: with softmax.c's backward unrun, the select-one
 * backward assertions (d_x_0 = sm_0(1-sm_0), d_x_i = -sm_0 sm_i for i>0)
 * fail; the numerical-stability forward (x ~ 1e3) would also NaN/Inf without
 * the max-subtract.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE
/* Forward: y = exp(x - max) / sum. Values match the stable softmax and the
   row sums to 1. */
Test(softmax_cov, softmax_1d_forward) {
	double xd[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle x = tensor_create(xd, s, 1, 0);
	TensorHandle y = tensor_softmax(x, 0);

	double e0 = exp(1.0 - 3.0), e1 = exp(2.0 - 3.0), e2 = exp(3.0 - 3.0);
	double se = e0 + e1 + e2;
	double out[3];
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], e0 / se, TEST_TOL_RELAXED, "softmax[0] (got %.12f)", out[0]);
	cr_assert_float_eq(out[1], e1 / se, TEST_TOL_RELAXED, "softmax[1] (got %.12f)", out[1]);
	cr_assert_float_eq(out[2], e2 / se, TEST_TOL_RELAXED, "softmax[2] (got %.12f)", out[2]);
	cr_assert_float_eq(out[0] + out[1] + out[2], 1.0, TEST_TOL_RELAXED, "softmax row sums to 1");
}

/* Backward with a non-uniform upstream gradient: loss = softmax(x)[0]
   (selected via narrow+sum), so grad on r is delta_{j,0}. The Jacobian
   reduces to d_x_i = sm_0 * (delta_i0 - sm_i), which exercises the full
   dot-subtraction path (lines 93-101 of softmax.c) with g != const. */
Test(softmax_cov, softmax_1d_backward_select) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);
	TensorHandle y = tensor_softmax(x, 0);
	TensorHandle cell0 = tensor_narrow(y, 0, 0, 1); /* [1] -> softmax[0] */
	TensorHandle loss = tensor_sum(cell0);
	tensor_backward(loss);

	double e0 = exp(1.0 - 3.0), e1 = exp(2.0 - 3.0), e2 = exp(3.0 - 3.0);
	double se = e0 + e1 + e2;
	double sm0 = e0 / se, sm1 = e1 / se, sm2 = e2 / se;
	cr_assert_float_eq(param_grad_item_at(0, 0), sm0 * (1.0 - sm0), TEST_TOL_RELAXED,
	                   "d_x[0] should be sm0*(1-sm0) (got %.12f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), -sm0 * sm1, TEST_TOL_RELAXED,
	                   "d_x[1] should be -sm0*sm1 (got %.12f)", param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), -sm0 * sm2, TEST_TOL_RELAXED,
	                   "d_x[2] should be -sm0*sm2 (got %.12f)", param_grad_item_at(0, 2));
	param_clear();
}

/* Backward of sum(softmax(x)): grad on r is all-ones, so the Jacobian sums to
   zero for every element (softmax rows are constant-sum). Exercises the same
   backward with the uniform-grad reduction. */
Test(softmax_cov, softmax_1d_backward_sum_zero) {
	param_clear();
	double xd[] = {-1.0, 0.5, 2.0, 0.0};
	int s[] = {4};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);
	TensorHandle loss = tensor_sum(tensor_softmax(x, 0));
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, TEST_TOL_RELAXED,
		                   "d sum(softmax)/dx[%d] should be 0 (got %.12f)", i,
		                   param_grad_item_at(0, i));
	param_clear();
}

/* Numerical-stability edge: large logits (~1e3) would overflow exp() without
   the max-subtract. Result must be finite, sum to 1, and equal the shifted
   softmax exp(x - max)/sum. */
Test(softmax_cov, softmax_1d_numerical_stability) {
	double xd[] = {1000.0, 1001.0, 1002.0};
	int s[] = {3};
	TensorHandle x = tensor_create(xd, s, 1, 0);
	double out[3];
	tensor_to_doubles(tensor_softmax(x, 0), out);

	double e0 = exp(-2.0), e1 = exp(-1.0), e2 = exp(0.0);
	double se = e0 + e1 + e2;
	for (int i = 0; i < 3; i++)
		cr_assert(isfinite(out[i]), "softmax[%d] must be finite (got %.6g)", i, out[i]);
	cr_assert_float_eq(out[0], e0 / se, TEST_TOL_RELAXED, "stable softmax[0] (got %.12f)", out[0]);
	cr_assert_float_eq(out[1], e1 / se, TEST_TOL_RELAXED, "stable softmax[1] (got %.12f)", out[1]);
	cr_assert_float_eq(out[2], e2 / se, TEST_TOL_RELAXED, "stable softmax[2] (got %.12f)", out[2]);
	cr_assert_float_eq(out[0] + out[1] + out[2], 1.0, TEST_TOL_RELAXED, "stable softmax sums to 1");
}

/* Rank-0 (scalar) forward arm: softmax of a single element is always 1.0.
   Exercises the `t->rank == 0` make_scalar branch of tensor_softmax. */
Test(softmax_cov, softmax_scalar_is_one) {
	TensorHandle x = tensor_create_scalar(5.0, 0);
	TensorHandle y = tensor_softmax(x, 0);
	cr_assert_float_eq(tensor_item(y), 1.0, TEST_TOL_RELAXED,
	                   "softmax(scalar) should be 1 (got %.12f)", tensor_item(y));
}
#endif /* BACKEND_TAPE */

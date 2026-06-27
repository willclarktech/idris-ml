/* Criterion suites for tensor_softmax_3d (forward + backward).
 *
 *   softmax along last dim of [B, m, n]:
 *     y[b, i, j] = exp(x[b, i, j]) / sum_k exp(x[b, i, k])
 *
 * Covers the F64 common path (suite `nn_softmax_softmax_3d`) plus the tape
 * F32 streamed forward/normalize/create arm (suite `softmax_3d_f32_cov`,
 * gated to BACKEND_TAPE).
 *
 * Construction note: there's no `tensor_create_3d_f64` FFI; build a
 * [1,1,N] via tensor_create_2d + tensor_reshape_3d. Reads use
 * tensor_to_doubles for flat-buffer extraction so we don't depend on
 * tensor_item_3d (which doesn't exist as an FFI).
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(nn_softmax_softmax_3d, forward_two_class) {
	/* x = [[[1, 2]]] shape [1,1,2] -> y = [[[1/(1+e), e/(1+e)]]]. */
	param_clear();
	double xd[] = {1.0, 2.0};
	TensorHandle flat = tensor_create_param_2d_f64(1, 2, hcopy(xd, 2));
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 1, 2);
	TensorHandle y = tensor_softmax_3d(x);
	double buf[2];
	tensor_to_doubles(y, buf);
	double denom = 1.0 + exp(1.0);
	cr_assert_float_eq(buf[0], 1.0 / denom, TEST_TOL_RELAXED,
	                   "softmax_3d[0,0,0] should be 1/(1+e) (got %.9f)", buf[0]);
	cr_assert_float_eq(buf[1], exp(1.0) / denom, TEST_TOL_RELAXED,
	                   "softmax_3d[0,0,1] should be e/(1+e) (got %.9f)", buf[1]);
}

Test(nn_softmax_softmax_3d, rows_sum_to_one) {
	/* x = [[[2.0, 3.0, 4.0]]] -> row should sum to 1. */
	param_clear();
	double xd[] = {2.0, 3.0, 4.0};
	TensorHandle flat = tensor_create_param_2d_f64(1, 3, hcopy(xd, 3));
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 1, 3);
	TensorHandle y = tensor_softmax_3d(x);
	double buf[3];
	tensor_to_doubles(y, buf);
	double total = buf[0] + buf[1] + buf[2];
	cr_assert_float_eq(total, 1.0, TEST_TOL_RELAXED,
	                   "softmax_3d row should sum to 1 (got %.9f, individual: %.6f %.6f %.6f)",
	                   total, buf[0], buf[1], buf[2]);
}

Test(nn_softmax_softmax_3d, backward_runs) {
	/* Loss = sum(softmax_3d(x)) = 1 (since the row sums to 1 always),
	 * so dL/dx = 0 everywhere (any change in x preserves the sum-to-1).
	 * This is a trivial backward but exercises the OP_SOFTMAX_3D dispatch. */
	param_clear();
	double xd[] = {1.0, 2.0};
	TensorHandle flat = tensor_create_param_2d_f64(1, 2, hcopy(xd, 2));
	param_register("x_flat", flat);
	TensorHandle x = tensor_reshape_3d(flat, 1, 1, 2);
	TensorHandle y = tensor_softmax_3d(x);
	TensorHandle loss = tensor_sum(y);
	cr_assert_float_eq(tensor_item(loss), 1.0, TEST_TOL_RELAXED,
	                   "sum(softmax_3d(x)) should be 1 (got %.9f)", tensor_item(loss));
	tensor_backward(loss);
	/* For loss = sum_j softmax(x)_j, gradient at every j is 0
	 * (sum-to-1 is invariant to x). */
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TEST_TOL_RELAXED,
	                   "grad x[0] for sum-loss should be 0 (got %.9f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, TEST_TOL_RELAXED,
	                   "grad x[1] for sum-loss should be 0 (got %.9f)", param_grad_item_at(0, 1));
}

#ifdef BACKEND_TAPE
/* Dtag mirroring DType.Core ("13/14/15=F16/F32/F64"). */
#define DTAG_F32 14

/* [B,m,n] = [2,1,2], every element equal (1.0). Each length-2 row softmaxes
   to [0.5, 0.5] — exact in F32. Drives the F32 store/normalize/create arm.
   Also runs the backward: for a uniform softmax under a sum-loss, dot==1 per
   row so every input grad is exactly 0 (line 49 tape_append + backward). */
Test(softmax_3d_f32_cov, f32_forward_backward_uniform) {
	param_clear();
	double in[4] = {1.0, 1.0, 1.0, 1.0};
	int sh[3] = {2, 1, 2};
	TensorHandle x = tensor_create_streamed(hcopy(in, 4), sh, 3, 1, 0, DTAG_F32);
	param_register("x", x);

	TensorHandle r = tensor_softmax_3d(x);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], 0.5, TEST_TOL_RELAXED, "softmax[%d] should be 0.5 (got %.6f)", i,
		                   out[i]);

	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, TEST_TOL_RELAXED,
		                   "uniform softmax grad[%d] should be 0 (got %.6f)", i,
		                   param_grad_item_at(0, i));

	param_clear();
}

/* [B,m,n] = [1,2,2] with two non-uniform rows. Row0 = [0, ln3]: after
   max-subtraction the exps are 1/3 and 1, sum 4/3, normalizing to
   [1/4, 3/4]. Row1 = [ln3, 0] -> [3/4, 1/4]. Confirms the F32 normalize
   produces the right values, not just the trivial uniform case. */
Test(softmax_3d_f32_cov, f32_forward_nonuniform) {
	double l3 = log(3.0);
	double in[4] = {0.0, l3, l3, 0.0};
	int sh[3] = {1, 2, 2};
	TensorHandle x = tensor_create_streamed(hcopy(in, 4), sh, 3, 0, 0, DTAG_F32);

	TensorHandle r = tensor_softmax_3d(x);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected[4] = {0.25, 0.75, 0.75, 0.25};
	/* exp(ln3) is not exact in single precision, so use an F32-scale tolerance
	   (1e-5) rather than TEST_TOL_RELAXED (1e-10 on tape). */
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected[i], 1e-5, "softmax[%d] should be %.2f (got %.6f)", i,
		                   expected[i], out[i]);
}
#endif

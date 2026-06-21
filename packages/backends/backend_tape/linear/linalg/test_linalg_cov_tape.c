/* Criterion suite `linalg_cov` — coverage top-up for the tape linalg dir.
 *
 * The pre-existing per-op suites (test_mv.c, test_dot_outer.c, ...) cover the
 * F64 happy paths. This file closes the remaining uncovered arms in the
 * sibling sources of this directory:
 *
 *   - mv.c          : F32 forward + F32 backward arms (tensor_mv_f32 +
 *                     tape_backward_mv DT_F32 branches), F64 zero-dim guard.
 *   - linear.c      : F64 forward+backward incl. bias-grad arm, F32 path +
 *                     F32 backward arms, F64 zero-dim (n=0) bias-only branch.
 *   - linear_2d.c   : F64 forward+backward incl. bias-grad arm, F32 path +
 *                     F32 backward arms, F64 zero-dim (i=0) bias-broadcast.
 *
 * Oracles are computed by hand from the inputs. F32-tagged tensors store as
 * float, so reads use TEST_TOL_RELAXED (all chosen values are exact integers
 * representable in single precision, so this is generous).
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

/* ------------------------------------------------------------------ mv.c */

Test(linalg_cov, mv_f32_forward_backward) {
	/* M=[[1,2,3],[4,5,6]] (F32), v=[7,8,9] (F32). M @ v = [50, 122].
	   loss=sum -> dM[i,j]=v[j]=[7,8,9] per row; dv[j]=sum_i M[i,j]=[5,7,9].
	   Hits tensor_mv_f32 + the DT_F32 backward arms of tape_backward_mv. */
	param_clear();
	double md[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double vd[] = {7.0, 8.0, 9.0};
	TensorHandle mat = tensor_create_2d_streamed(2, 3, hcopy(md, 6), 1, 0, 14);
	TensorHandle vec = tensor_create_1d_streamed(3, hcopy(vd, 3), 1, 0, 14);
	param_register("mat", mat);
	param_register("vec", vec);
	TensorHandle r = tensor_mv(mat, vec);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 50.0, TEST_TOL_RELAXED, "mv_f32[0] should be 50 (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], 122.0, TEST_TOL_RELAXED, "mv_f32[1] should be 122 (got %.6f)",
	                   out[1]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expected_M[] = {7, 8, 9, 7, 8, 9};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected_M[i], TEST_TOL_RELAXED,
		                   "dM[%d] should be %.1f (got %.6f)", i, expected_M[i],
		                   param_grad_item_at(0, i));
	double expected_v[] = {5, 7, 9};
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), expected_v[j], TEST_TOL_RELAXED,
		                   "dv[%d] should be %.1f (got %.6f)", j, expected_v[j],
		                   param_grad_item_at(1, j));
	param_clear();
}

Test(linalg_cov, mv_zero_dim_n0) {
	/* mat=[2,0], vec=[0]: the m==0||n==0 guard returns a zero tensor of
	   shape [m]=[2]. Covers the F64 zero-dim branch of tensor_mv. */
	double md[] = {0};
	double vd[] = {0};
	int sm[] = {2, 0};
	int sv[] = {0};
	TensorHandle mat = tensor_create(md, sm, 2, 0);
	TensorHandle vec = tensor_create(vd, sv, 1, 0);
	TensorHandle r = tensor_mv(mat, vec);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_TIGHT);
	cr_assert_float_eq(out[1], 0.0, TEST_TOL_TIGHT);
}

/* -------------------------------------------------------------- linear.c */

Test(linalg_cov, linear_f64_forward_backward_bias) {
	/* y = W @ x + b, W=[[1,2,3],[4,5,6]], x=[7,8,9], b=[10,20].
	   y = [50+10, 122+20] = [60, 142]. loss=sum:
	   dW[i,j]=x[j]=[7,8,9] per row; dx[j]=sum_i W[i,j]=[5,7,9]; db[i]=1.
	   Covers tensor_linear F64 forward + all three backward arms (incl bias). */
	param_clear();
	double wd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double xd[] = {7.0, 8.0, 9.0};
	double bd[] = {10.0, 20.0};
	int sw[] = {2, 3};
	int sx[] = {3};
	int sb[] = {2};
	TensorHandle W = tensor_create(wd, sw, 2, 1);
	TensorHandle x = tensor_create(xd, sx, 1, 1);
	TensorHandle bias = tensor_create(bd, sb, 1, 1);
	param_register("W", W);
	param_register("x", x);
	param_register("bias", bias);
	TensorHandle r = tensor_linear(W, x, bias);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 60.0, TEST_TOL_TIGHT, "y[0] should be 60 (got %.9f)", out[0]);
	cr_assert_float_eq(out[1], 142.0, TEST_TOL_TIGHT, "y[1] should be 142 (got %.9f)", out[1]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expected_W[] = {7, 8, 9, 7, 8, 9};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected_W[i], TEST_TOL_TIGHT,
		                   "dW[%d] should be %.1f (got %.9f)", i, expected_W[i],
		                   param_grad_item_at(0, i));
	double expected_x[] = {5, 7, 9};
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), expected_x[j], TEST_TOL_TIGHT,
		                   "dx[%d] should be %.1f (got %.9f)", j, expected_x[j],
		                   param_grad_item_at(1, j));
	cr_assert_float_eq(param_grad_item_at(2, 0), 1.0, TEST_TOL_TIGHT, "db[0] should be 1");
	cr_assert_float_eq(param_grad_item_at(2, 1), 1.0, TEST_TOL_TIGHT, "db[1] should be 1");
	param_clear();
}

Test(linalg_cov, linear_f32_forward_backward_bias) {
	/* Same numbers as the F64 case, but F32-tagged: covers
	   tensor_linear_f32 + the DT_F32 backward arms (dW loop, dx loop) and
	   the bias-grad accumulate. */
	param_clear();
	double wd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double xd[] = {7.0, 8.0, 9.0};
	double bd[] = {10.0, 20.0};
	TensorHandle W = tensor_create_2d_streamed(2, 3, hcopy(wd, 6), 1, 0, 14);
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(xd, 3), 1, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(2, hcopy(bd, 2), 1, 0, 14);
	param_register("W", W);
	param_register("x", x);
	param_register("bias", bias);
	TensorHandle r = tensor_linear(W, x, bias);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 60.0, TEST_TOL_RELAXED, "y_f32[0] should be 60 (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], 142.0, TEST_TOL_RELAXED, "y_f32[1] should be 142 (got %.6f)",
	                   out[1]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expected_W[] = {7, 8, 9, 7, 8, 9};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected_W[i], TEST_TOL_RELAXED,
		                   "dW_f32[%d] should be %.1f (got %.6f)", i, expected_W[i],
		                   param_grad_item_at(0, i));
	double expected_x[] = {5, 7, 9};
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), expected_x[j], TEST_TOL_RELAXED,
		                   "dx_f32[%d] should be %.1f (got %.6f)", j, expected_x[j],
		                   param_grad_item_at(1, j));
	cr_assert_float_eq(param_grad_item_at(2, 0), 1.0, TEST_TOL_RELAXED, "db_f32[0] should be 1");
	cr_assert_float_eq(param_grad_item_at(2, 1), 1.0, TEST_TOL_RELAXED, "db_f32[1] should be 1");
	param_clear();
}

Test(linalg_cov, linear_zero_dim_n0_bias) {
	/* W=[2,0], x=[0], bias=[10,20]: the n==0 guard drops the matmul and the
	   output is just the bias. Covers the F64 zero-dim bias-only branch. */
	double wd[] = {0};
	double xd[] = {0};
	double bd[] = {10.0, 20.0};
	int sw[] = {2, 0};
	int sx[] = {0};
	int sb[] = {2};
	TensorHandle W = tensor_create(wd, sw, 2, 0);
	TensorHandle x = tensor_create(xd, sx, 1, 0);
	TensorHandle bias = tensor_create(bd, sb, 1, 0);
	TensorHandle r = tensor_linear(W, x, bias);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 10.0, TEST_TOL_TIGHT, "y[0] should be bias 10 (got %.9f)", out[0]);
	cr_assert_float_eq(out[1], 20.0, TEST_TOL_TIGHT, "y[1] should be bias 20 (got %.9f)", out[1]);
}

/* ----------------------------------------------------------- linear_2d.c */

Test(linalg_cov, linear_2d_f64_forward_backward_bias) {
	/* Y[B,o] = X[B,i] @ W[o,i]^T + bias[o].
	   W=[[1,2,3],[4,5,6]] (o=2,i=3), X=[[1,1,1],[2,2,2]] (B=2), bias=[10,20].
	   Y = [[6+10, 15+20], [12+10, 30+20]] = [[16,35],[22,50]].
	   loss=sum (dY all ones):
	     dW[o,j]=sum_b X[b,j]=3 (all); dX[b,j]=sum_o W[o,j]=[5,7,9]; db[o]=B=2.
	   Covers tensor_linear_2d F64 forward + all three backward arms. */
	param_clear();
	double wd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double xd[] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
	double bd[] = {10.0, 20.0};
	int sw[] = {2, 3};
	int sx[] = {2, 3};
	int sb[] = {2};
	TensorHandle W = tensor_create(wd, sw, 2, 1);
	TensorHandle X = tensor_create(xd, sx, 2, 1);
	TensorHandle bias = tensor_create(bd, sb, 1, 1);
	param_register("W", W);
	param_register("X", X);
	param_register("bias", bias);
	TensorHandle r = tensor_linear_2d(W, X, bias);
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected_Y[] = {16, 35, 22, 50};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected_Y[i], TEST_TOL_TIGHT, "Y[%d] should be %.1f (got %.9f)",
		                   i, expected_Y[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 3.0, TEST_TOL_TIGHT,
		                   "dW[%d] should be 3 (got %.9f)", i, param_grad_item_at(0, i));
	double expected_X[] = {5, 7, 9, 5, 7, 9};
	for (int j = 0; j < 6; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), expected_X[j], TEST_TOL_TIGHT,
		                   "dX[%d] should be %.1f (got %.9f)", j, expected_X[j],
		                   param_grad_item_at(1, j));
	cr_assert_float_eq(param_grad_item_at(2, 0), 2.0, TEST_TOL_TIGHT, "db[0] should be 2");
	cr_assert_float_eq(param_grad_item_at(2, 1), 2.0, TEST_TOL_TIGHT, "db[1] should be 2");
	param_clear();
}

Test(linalg_cov, linear_2d_f32_forward_backward_bias) {
	/* Same numbers, F32-tagged: covers the DT_F32 forward (sgemm) + the
	   DT_F32 backward arms (dW, dX loops) and the bias-grad accumulate. */
	param_clear();
	double wd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double xd[] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
	double bd[] = {10.0, 20.0};
	TensorHandle W = tensor_create_2d_streamed(2, 3, hcopy(wd, 6), 1, 0, 14);
	TensorHandle X = tensor_create_2d_streamed(2, 3, hcopy(xd, 6), 1, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(2, hcopy(bd, 2), 1, 0, 14);
	param_register("W", W);
	param_register("X", X);
	param_register("bias", bias);
	TensorHandle r = tensor_linear_2d(W, X, bias);
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected_Y[] = {16, 35, 22, 50};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected_Y[i], TEST_TOL_RELAXED,
		                   "Y_f32[%d] should be %.1f (got %.6f)", i, expected_Y[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 3.0, TEST_TOL_RELAXED,
		                   "dW_f32[%d] should be 3 (got %.6f)", i, param_grad_item_at(0, i));
	double expected_X[] = {5, 7, 9, 5, 7, 9};
	for (int j = 0; j < 6; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), expected_X[j], TEST_TOL_RELAXED,
		                   "dX_f32[%d] should be %.1f (got %.6f)", j, expected_X[j],
		                   param_grad_item_at(1, j));
	cr_assert_float_eq(param_grad_item_at(2, 0), 2.0, TEST_TOL_RELAXED, "db_f32[0] should be 2");
	cr_assert_float_eq(param_grad_item_at(2, 1), 2.0, TEST_TOL_RELAXED, "db_f32[1] should be 2");
	param_clear();
}

Test(linalg_cov, linear_2d_zero_dim_i0_bias) {
	/* W=[2,0], X=[2,0], bias=[10,20]: the ii==0 guard drops the matmul; the
	   bias broadcasts across the batch. Covers the F64 zero-dim broadcast. */
	double wd[] = {0};
	double xd[] = {0};
	double bd[] = {10.0, 20.0};
	int sw[] = {2, 0};
	int sx[] = {2, 0};
	int sb[] = {2};
	TensorHandle W = tensor_create(wd, sw, 2, 0);
	TensorHandle X = tensor_create(xd, sx, 2, 0);
	TensorHandle bias = tensor_create(bd, sb, 1, 0);
	TensorHandle r = tensor_linear_2d(W, X, bias);
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected_Y[] = {10, 20, 10, 20};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected_Y[i], TEST_TOL_TIGHT,
		                   "Y[%d] should be broadcast bias %.1f (got %.9f)", i, expected_Y[i],
		                   out[i]);
}

/* Criterion suite `matmul_f32_cov` — F32-tagged coverage for tensor_matmul's
 * rank-dispatch tail in matmul.c: the 2D×1D arm (delegates to tensor_mv) and
 * the 2D×2D else arm (elementwise tensor_mul fallback).
 *
 * Tape-only: bare F32 creation aborts on tape, so F32 tensors are built via
 * the streamed dtag-14 creators; the file compiles into every backend's test
 * binary, hence the BACKEND_TAPE guard.
 */

#ifdef BACKEND_TAPE

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

/* ------------------------------------------------ line 54: 2D × 1D -> mv */

Test(matmul_f32_cov, matmul_2d_1d_dispatches_to_mv_f32) {
	/* M=[[1,2,3],[4,5,6]] (F32), v=[7,8,9] (F32). tensor_matmul sees
	   rank(M)==2, rank(v)==1 and forwards to tensor_mv -> M @ v = [50, 122].
	   loss=sum -> dM[i,j]=v[j]=[7,8,9] per row; dv[j]=sum_i M[i,j]=[5,7,9].
	   Covers line 54 (the 2D×1D delegate). */
	param_clear();
	double md[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double vd[] = {7.0, 8.0, 9.0};
	TensorHandle mat = tensor_create_2d_streamed(2, 3, hcopy(md, 6), 1, 0, 14);
	TensorHandle vec = tensor_create_1d_streamed(3, hcopy(vd, 3), 1, 0, 14);
	param_register("mat", mat);
	param_register("vec", vec);
	TensorHandle r = tensor_matmul(mat, vec);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 50.0, TEST_TOL_RELAXED, "matmul2d1d[0] should be 50 (got %.6f)",
	                   out[0]);
	cr_assert_float_eq(out[1], 122.0, TEST_TOL_RELAXED, "matmul2d1d[1] should be 122 (got %.6f)",
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

/* --------------------------------------- line 55: 2D × 2D -> mul fallback */

Test(matmul_f32_cov, matmul_2d_2d_dispatches_to_mul_f32) {
	/* A=[[1,2],[3,4]] (F32), B=[[5,6],[7,8]] (F32). Neither the 1D×2D nor the
	   2D×1D arm matches, so tensor_matmul falls through to the elementwise
	   tensor_mul fallback: A*B = [[5,12],[21,32]] (flat [5,12,21,32]).
	   loss=sum -> dA=B=[5,6,7,8]; dB=A=[1,2,3,4]. Covers line 55. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0};
	double bd[] = {5.0, 6.0, 7.0, 8.0};
	TensorHandle A = tensor_create_2d_streamed(2, 2, hcopy(ad, 4), 1, 0, 14);
	TensorHandle B = tensor_create_2d_streamed(2, 2, hcopy(bd, 4), 1, 0, 14);
	param_register("A", A);
	param_register("B", B);
	TensorHandle r = tensor_matmul(A, B);
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected_r[] = {5, 12, 21, 32};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected_r[i], TEST_TOL_RELAXED,
		                   "matmul2d2d[%d] should be %.1f (got %.6f)", i, expected_r[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expected_A[] = {5, 6, 7, 8};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected_A[i], TEST_TOL_RELAXED,
		                   "dA[%d] should be %.1f (got %.6f)", i, expected_A[i],
		                   param_grad_item_at(0, i));
	double expected_B[] = {1, 2, 3, 4};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), expected_B[i], TEST_TOL_RELAXED,
		                   "dB[%d] should be %.1f (got %.6f)", i, expected_B[i],
		                   param_grad_item_at(1, i));
	param_clear();
}

#endif /* BACKEND_TAPE */

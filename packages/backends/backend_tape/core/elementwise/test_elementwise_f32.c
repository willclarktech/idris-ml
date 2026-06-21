/* F32 elementwise arm coverage (tape): broadcast div, sigmoid scalar kernel, and
 * the vForce unary cases. F32 on tape is built via the streamed dtag-14 creators.
 */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* Broadcast F32 div ([2,3] / [3]) -> scalar_fn = fn_div_f32 (the matching-shape
   path uses vDSP and would skip it). */
Test(elementwise_f32_cov, div_broadcast) {
	double ad[] = {2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
	double bd[] = {2.0, 4.0, 8.0};
	int sa[] = {2, 3};
	TensorHandle a = tensor_create_2d_streamed(2, 3, hcopy(ad, 6), 0, 0, 14);
	(void)sa;
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(bd, 3), 0, 0, 14);
	TensorHandle r = tensor_div(a, b); /* broadcast -> fn_div_f32 */
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "got %s", tensor_dtype_name(r));
	double out[6];
	tensor_to_doubles(r, out);
	double exp[] = {1, 1, 1, 8, 8, 8};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], exp[i], TEST_TOL_RELAXED, "div_bcast[%d] (got %.6f)", i, out[i]);
}

/* F32 sigmoid: not a vForce op, so the F32 scalar kernel (fn_sigmoid_f32) runs. */
Test(elementwise_f32_cov, sigmoid) {
	double d[] = {0.0, 0.0};
	TensorHandle x = tensor_create_1d_streamed(2, hcopy(d, 2), 0, 0, 14);
	TensorHandle r = tensor_sigmoid(x);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "got %s", tensor_dtype_name(r));
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 0.5, TEST_TOL_RELAXED, "sigmoid(0) (got %.6f)", out[0]);
}

/* F32 vForce unary ops (matching shape) — drive the VV_* cases in _kernels.inc. */
Test(elementwise_f32_cov, vforce_unary) {
	double d[] = {1.0, 4.0};
	TensorHandle x1 = tensor_create_1d_streamed(2, hcopy(d, 2), 0, 0, 14);
	TensorHandle lg = tensor_log(x1);
	cr_assert_float_eq(tensor_item_1d(lg, 0), 0.0, 1e-5, "log(1) (got %.6f)",
	                   tensor_item_1d(lg, 0));
	TensorHandle x2 = tensor_create_1d_streamed(2, hcopy(d, 2), 0, 0, 14);
	TensorHandle sq = tensor_sqrt(x2);
	cr_assert_float_eq(tensor_item_1d(sq, 1), 2.0, 1e-5, "sqrt(4) (got %.6f)",
	                   tensor_item_1d(sq, 1));
	double dn[] = {-3.0, 4.0};
	TensorHandle x3 = tensor_create_1d_streamed(2, hcopy(dn, 2), 0, 0, 14);
	TensorHandle ab = tensor_abs(x3);
	cr_assert_float_eq(tensor_item_1d(ab, 0), 3.0, 1e-5, "abs(-3) (got %.6f)",
	                   tensor_item_1d(ab, 0));
	TensorHandle x4 = tensor_create_1d_streamed(2, hcopy(d, 2), 0, 0, 14);
	TensorHandle th = tensor_tanh(x4);
	cr_assert_float_eq(tensor_item_1d(th, 0), 0.7615941559, 1e-5, "tanh(1) (got %.6f)",
	                   tensor_item_1d(th, 0));
}

#endif /* BACKEND_TAPE */

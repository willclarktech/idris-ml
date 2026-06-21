/* Coverage suite `layer_norm_2d_cov` — closes the F32 arms of tape
 * `tensor_layer_norm_2d` (nn/norm/layer_norm_2d.c).
 *
 * The base suite covers the F64 path and the F32 grad-tracking path. The
 * remaining uncovered arm is the F32 *no-grad* tail (lines 66-68): when the
 * result does not require grad, the F32 branch frees the cached x_hat / rstd
 * buffers instead of stashing them in a LayerNormMeta. A grad-free F32 call
 * is the only way to reach those frees, so the first test drives exactly
 * that. The second test re-exercises the F32 grad path forward+backward with
 * a hand-computed oracle.
 *
 * (Line 24 — the mixed-dtype abort guard — is an abort path covered by the
 * dedicated death/guard suites and is intentionally NOT retested here.)
 *
 * Tape aborts on bare tensor_create_*_f32, so F32 tensors are built via the
 * streamed dtag-14 creators (which own + free their hcopy'd buffer). Inputs
 * are chosen so normalization is exact in single precision: each row is
 * antisymmetric, so mean=0 and x_hat = [-1, +1] exactly, with rstd = 1 and
 * 1/2 (both exact).
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* x = [[-1, 1], [-2, 2]] (F32), gamma = [2, 1], bias = [10, 20], eps = 0.
   Row0: mean 0, var 1, rstd 1, x_hat [-1, 1].
   Row1: mean 0, var 4, rstd 1/2, x_hat [-1, 1].
   out[i] = gamma .* x_hat[i] + bias = [2*-1+10, 1*1+20] = [8, 21] for both rows.
   requires_grad = 0 on all inputs -> rg = 0 -> the F32 no-grad else branch
   frees x_hat / rstd (the target arm). */
Test(layer_norm_2d_cov, f32_forward_no_grad) {
	param_clear();
	double xd[] = {-1.0, 1.0, -2.0, 2.0};
	double gd[] = {2.0, 1.0};
	double bd[] = {10.0, 20.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), 0, 0, 14);
	TensorHandle gamma = tensor_create_1d_streamed(2, hcopy(gd, 2), 0, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(2, hcopy(bd, 2), 0, 0, 14);

	TensorHandle r = tensor_layer_norm_2d(x, gamma, bias, 0.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected[] = {8.0, 21.0, 8.0, 21.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED,
		                   "ln2d_f32 nograd out[%d] should be %.1f (got %.6f)", i, expected[i],
		                   out[i]);
	param_clear();
}

/* Same inputs, requires_grad = 1 on all three -> the F32 grad path (caches
   meta) + the layer-norm backward. With a sum loss (upstream grad all ones):
     dgamma[j] = sum_i x_hat[i,j]          = [-2, 2]
     dbias[j]  = sum_i 1                    = [2, 2]
     dinput    = 0 everywhere (layer norm output is invariant to the input's
                 mean/scale, so a uniform upstream grad maps to a zero input
                 grad). */
Test(layer_norm_2d_cov, f32_forward_backward_grad) {
	param_clear();
	double xd[] = {-1.0, 1.0, -2.0, 2.0};
	double gd[] = {2.0, 1.0};
	double bd[] = {10.0, 20.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), 1, 0, 14);
	TensorHandle gamma = tensor_create_1d_streamed(2, hcopy(gd, 2), 1, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(2, hcopy(bd, 2), 1, 0, 14);
	param_register("x", x);
	param_register("gamma", gamma);
	param_register("bias", bias);

	TensorHandle r = tensor_layer_norm_2d(x, gamma, bias, 0.0);
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected[] = {8.0, 21.0, 8.0, 21.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED,
		                   "ln2d_f32 grad out[%d] should be %.1f (got %.6f)", i, expected[i],
		                   out[i]);

	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);

	/* dinput = 0 everywhere. */
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, TEST_TOL_RELAXED,
		                   "dx[%d] should be 0 (got %.6f)", i, param_grad_item_at(0, i));
	/* dgamma = [-2, 2]. */
	double expected_g[] = {-2.0, 2.0};
	for (int j = 0; j < 2; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), expected_g[j], TEST_TOL_RELAXED,
		                   "dgamma[%d] should be %.1f (got %.6f)", j, expected_g[j],
		                   param_grad_item_at(1, j));
	/* dbias = [2, 2]. */
	for (int j = 0; j < 2; j++)
		cr_assert_float_eq(param_grad_item_at(2, j), 2.0, TEST_TOL_RELAXED,
		                   "dbias[%d] should be 2 (got %.6f)", j, param_grad_item_at(2, j));
	param_clear();
}

#endif /* BACKEND_TAPE */

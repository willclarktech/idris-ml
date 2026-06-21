/* Criterion suite for tensor_rms_norm_2d (forward).
 *
 *   rms_norm_2d(input, weight, eps):
 *     For each row of input [M, N]:
 *       rstd_i = 1 / sqrt((1/N) sum_j input[i, j]^2 + eps)
 *       out[i, j] = input[i, j] * rstd_i * weight[j]
 *
 * Matches the HF LlamaRMSNorm formula (no centering, no bias).
 * Replaces the per-row 7-primitive chain in HfCommon.applyRmsNorm2dRaw
 * with one fused FFI call.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(nn_norm_rms_norm, forward_unit_weight) {
	/* Input: [[1, 2, 3, 4]], weight = [1, 1, 1, 1], eps = 1e-6.
	 * mean_sq = (1 + 4 + 9 + 16) / 4 = 7.5
	 * rstd    = 1 / sqrt(7.5 + 1e-6) ≈ 0.36514837...
	 * out[j]  = input[j] * rstd
	 */
	param_clear();
	double in_d[] = {1.0, 2.0, 3.0, 4.0};
	double w_d[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle input = tensor_create_2d_f64(1, 4, hcopy(in_d, 4), 0);
	TensorHandle weight = tensor_create_1d_f64(4, hcopy(w_d, 4), 0);
	double eps = 1e-6;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	double buf[4];
	tensor_to_doubles(r, buf);
	double rstd = 1.0 / sqrt(7.5 + eps);
	double expect[] = {1.0 * rstd, 2.0 * rstd, 3.0 * rstd, 4.0 * rstd};
	for (int j = 0; j < 4; j++) {
		cr_assert_float_eq(buf[j], expect[j], TEST_TOL_RELAXED,
		                   "rms_norm[%d] should be %.9f (got %.9f)", j, expect[j], buf[j]);
	}
}

Test(nn_norm_rms_norm, forward_per_row_independent) {
	/* Two rows. Each row normalized independently — different mean_sq
	 * per row should produce different rstd, scaling each row by its
	 * own factor.
	 *   row 0: [1, 1, 1, 1] -> mean_sq = 1.0, rstd = 1/sqrt(1+eps) ≈ 1
	 *   row 1: [2, 2, 2, 2] -> mean_sq = 4.0, rstd = 1/sqrt(4+eps) ≈ 0.5
	 * weight = [1, 1, 1, 1] keeps gain unity.
	 */
	param_clear();
	double in_d[] = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
	double w_d[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle input = tensor_create_2d_f64(2, 4, hcopy(in_d, 8), 0);
	TensorHandle weight = tensor_create_1d_f64(4, hcopy(w_d, 4), 0);
	double eps = 1e-6;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	double buf[8];
	tensor_to_doubles(r, buf);
	double rstd0 = 1.0 / sqrt(1.0 + eps);
	double rstd1 = 1.0 / sqrt(4.0 + eps);
	for (int j = 0; j < 4; j++) {
		cr_assert_float_eq(buf[j], 1.0 * rstd0, TEST_TOL_RELAXED, "row0[%d] expected %.9f got %.9f",
		                   j, 1.0 * rstd0, buf[j]);
		cr_assert_float_eq(buf[4 + j], 2.0 * rstd1, TEST_TOL_RELAXED,
		                   "row1[%d] expected %.9f got %.9f", j, 2.0 * rstd1, buf[4 + j]);
	}
}

Test(nn_norm_rms_norm, forward_weight_scaling) {
	/* Per-column weight applies after normalization.
	 * Input: [[3, 4]]  (mean_sq = 12.5, rstd = 1/sqrt(12.5+eps) ≈ 0.2828427)
	 * weight = [2, 3]
	 * out[0] = 3 * rstd * 2 = 6 * rstd
	 * out[1] = 4 * rstd * 3 = 12 * rstd
	 */
	param_clear();
	double in_d[] = {3.0, 4.0};
	double w_d[] = {2.0, 3.0};
	TensorHandle input = tensor_create_2d_f64(1, 2, hcopy(in_d, 2), 0);
	TensorHandle weight = tensor_create_1d_f64(2, hcopy(w_d, 2), 0);
	double eps = 1e-6;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	double buf[2];
	tensor_to_doubles(r, buf);
	double rstd = 1.0 / sqrt(12.5 + eps);
	cr_assert_float_eq(buf[0], 6.0 * rstd, TEST_TOL_RELAXED, "weighted[0] expected %.9f got %.9f",
	                   6.0 * rstd, buf[0]);
	cr_assert_float_eq(buf[1], 12.0 * rstd, TEST_TOL_RELAXED, "weighted[1] expected %.9f got %.9f",
	                   12.0 * rstd, buf[1]);
}

Test(nn_norm_rms_norm, forward_matches_decomposed_chain) {
	/* Strongest correctness check: fused op must match the same per-row
	 * formula computed via independent host-side math. Same shape that
	 * HfLlama hits at run-time (seq=4, hidden=8 — small enough to keep
	 * the F32 tolerance generous on mlx). Random-ish nonzero inputs.
	 */
	param_clear();
	double in_d[32];
	double w_d[8];
	for (int i = 0; i < 32; i++)
		in_d[i] = (i % 5 == 0) ? -0.7 : 0.3 + (i * 0.11);
	for (int j = 0; j < 8; j++)
		w_d[j] = 0.5 + j * 0.1;
	TensorHandle input = tensor_create_2d_f64(4, 8, hcopy(in_d, 32), 0);
	TensorHandle weight = tensor_create_1d_f64(8, hcopy(w_d, 8), 0);
	double eps = 1e-5;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	double got[32];
	tensor_to_doubles(r, got);
	for (int i = 0; i < 4; i++) {
		double s = 0;
		for (int j = 0; j < 8; j++)
			s += in_d[i * 8 + j] * in_d[i * 8 + j];
		double rstd = 1.0 / sqrt(s / 8.0 + eps);
		for (int j = 0; j < 8; j++) {
			double expect = in_d[i * 8 + j] * rstd * w_d[j];
			cr_assert_float_eq(got[i * 8 + j], expect, TEST_TOL_RELAXED,
			                   "rms_norm[%d,%d] expected %.9f got %.9f", i, j, expect,
			                   got[i * 8 + j]);
		}
	}
}

/* ---------------------------------------------------------------------- */
/* F32 forward path (rms_norm_2d.c lines 37-71). Tape supports F32; the    */
/* whole `if (t->dtype_tag == DT_F32)` block is a separate kernel that the */
/* F64 tests above never reach. F32 readback carries ~1e-6 error, so       */
/* assert at an explicit 1e-5 literal (NOT TEST_TOL_TIGHT).                */
/* ---------------------------------------------------------------------- */

Test(nn_norm_rms_norm, forward_f32_weight_scaling) {
	/* Mirror of forward_weight_scaling but on F32 tensors. Drives the
	 * F32 forward kernel (var accumulation, rstd, x_hat, weighted store),
	 * the make_tensor_arena_f32 result, and the dtype-name tag.
	 *   Input: [[3, 4]]  (mean_sq = 12.5, rstd = 1/sqrt(12.5+eps))
	 *   weight = [2, 3]
	 *   out[0] = 3 * rstd * 2 = 6 * rstd ; out[1] = 4 * rstd * 3 = 12 * rstd
	 */
	param_clear();
	double in_d[] = {3.0, 4.0};
	double w_d[] = {2.0, 3.0};
	TensorHandle input = tensor_create_2d_streamed(1, 2, hcopy(in_d, 2), 0, 0, 14);
	TensorHandle weight = tensor_create_1d_streamed(2, hcopy(w_d, 2), 0, 0, 14);
	double eps = 1e-6;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "result dtype should be F32 (got %s)",
	                 tensor_dtype_name(r));
	double buf[2];
	tensor_to_doubles(r, buf);
	double rstd = 1.0 / sqrt(12.5 + eps);
	cr_assert_float_eq(buf[0], 6.0 * rstd, 1e-5, "f32 weighted[0] expected %.9f got %.9f",
	                   6.0 * rstd, buf[0]);
	cr_assert_float_eq(buf[1], 12.0 * rstd, 1e-5, "f32 weighted[1] expected %.9f got %.9f",
	                   12.0 * rstd, buf[1]);
}

Test(nn_norm_rms_norm, forward_f32_per_row_independent) {
	/* Two F32 rows normalized independently — exercises the F32 outer
	 * row loop more than once and the per-row rstd divergence.
	 *   row 0: [1, 1, 1, 1] -> mean_sq = 1.0, rstd = 1/sqrt(1+eps)
	 *   row 1: [2, 2, 2, 2] -> mean_sq = 4.0, rstd = 1/sqrt(4+eps)
	 */
	param_clear();
	double in_d[] = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
	double w_d[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle input = tensor_create_2d_streamed(2, 4, hcopy(in_d, 8), 0, 0, 14);
	TensorHandle weight = tensor_create_1d_streamed(4, hcopy(w_d, 4), 0, 0, 14);
	double eps = 1e-6;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	double buf[8];
	tensor_to_doubles(r, buf);
	double rstd0 = 1.0 / sqrt(1.0 + eps);
	double rstd1 = 1.0 / sqrt(4.0 + eps);
	for (int j = 0; j < 4; j++) {
		cr_assert_float_eq(buf[j], 1.0 * rstd0, 1e-5, "f32 row0[%d] expected %.9f got %.9f", j,
		                   1.0 * rstd0, buf[j]);
		cr_assert_float_eq(buf[4 + j], 2.0 * rstd1, 1e-5, "f32 row1[%d] expected %.9f got %.9f", j,
		                   2.0 * rstd1, buf[4 + j]);
	}
}

Test(nn_norm_rms_norm, backward_f32_input_and_weight_grads) {
	/* F32 forward with requires_grad -> drives the F32 rg=1 branch
	 * (lines 57-69: meta alloc, tape_append, x_hat/rstd cached). Backward
	 * reads x_hat/rstd uniformly as double*, so grads match the F64 case.
	 *   Single row [[3, 4]], weight = [1, 1]; loss = sum(out); dout = ones.
	 *   d(x)[0] = rstd*(1 - 10.5*rstd^2), d(x)[1] = rstd*(1 - 14.0*rstd^2)
	 *   d(w)[j] = x_hat[j] = {3*rstd, 4*rstd}
	 */
	param_clear();
	double in_d[] = {3.0, 4.0};
	double w_d[] = {1.0, 1.0};
	TensorHandle input = tensor_create_2d_streamed(1, 2, hcopy(in_d, 2), 1, 0, 14);
	TensorHandle weight = tensor_create_1d_streamed(2, hcopy(w_d, 2), 1, 0, 14);
	param_register("input", input);   /* param_idx 0 */
	param_register("weight", weight); /* param_idx 1 */
	double eps = 1e-9;
	TensorHandle loss = tensor_sum(tensor_rms_norm_2d(input, weight, eps));
	tensor_backward(loss);

	double rstd = 1.0 / sqrt(12.5 + eps);
	double r2 = rstd * rstd;
	double dx0 = rstd * (1.0 - 10.5 * r2);
	double dx1 = rstd * (1.0 - 14.0 * r2);
	cr_assert_float_eq(param_grad_item_at(0, 0), dx0, 1e-5, "f32 d(x)[0] expected %.9f got %.9f",
	                   dx0, param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), dx1, 1e-5, "f32 d(x)[1] expected %.9f got %.9f",
	                   dx1, param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(1, 0), 3.0 * rstd, 1e-5,
	                   "f32 d(w)[0] expected %.9f got %.9f", 3.0 * rstd, param_grad_item_at(1, 0));
	cr_assert_float_eq(param_grad_item_at(1, 1), 4.0 * rstd, 1e-5,
	                   "f32 d(w)[1] expected %.9f got %.9f", 4.0 * rstd, param_grad_item_at(1, 1));
}

Test(nn_norm_rms_norm, forward_f32_no_grad_frees_meta) {
	/* F32 forward with requires_grad = 0 -> drives the F32 else branch
	 * (lines 67-69: free(x_hat); free(rstd)) where no tape entry is made.
	 * Just check the value is correct and no crash on the free path.
	 */
	param_clear();
	double in_d[] = {1.0, 2.0, 3.0, 4.0};
	double w_d[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle input = tensor_create_2d_streamed(1, 4, hcopy(in_d, 4), 0, 0, 14);
	TensorHandle weight = tensor_create_1d_streamed(4, hcopy(w_d, 4), 0, 0, 14);
	double eps = 1e-6;
	TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
	double buf[4];
	tensor_to_doubles(r, buf);
	double rstd = 1.0 / sqrt(7.5 + eps);
	for (int j = 0; j < 4; j++)
		cr_assert_float_eq(buf[j], (j + 1.0) * rstd, 1e-5, "f32 nograd[%d] expected %.9f got %.9f",
		                   j, (j + 1.0) * rstd, buf[j]);
}

/* ---------------------------------------------------------------------- */
/* Backward (OP_RMS_NORM_2D). Covers the rg=1 tape-append branch and the  */
/* tape_backward_rms_norm_2d body: both the weight-grad and input-grad    */
/* accumulation paths.                                                    */
/* ---------------------------------------------------------------------- */

Test(nn_norm_rms_norm, backward_input_and_weight_grads) {
	/* Single row [[3, 4]], weight = [1, 1], eps -> 0; loss = sum(out).
	 *   mean_sq = 12.5, rstd = 1/sqrt(12.5), x_hat = [3*rstd, 4*rstd]
	 *   dout = [1, 1] (sum reduction)
	 *
	 * d(weight)[j] = sum_i dout[i,j] * x_hat[i,j] = x_hat[j]
	 *   -> [3*rstd, 4*rstd]
	 *
	 * d(x)[j] = rstd * (dout_w[j] - x_hat[j] * (1/n) * sum_k dout_w[k]*x_hat[k])
	 *   dout_w = weight = [1, 1]
	 *   sum_k dout_w*x_hat = 7*rstd ; mean = 3.5*rstd
	 *   d(x)[0] = rstd * (1 - 3*rstd * 3.5*rstd) = rstd * (1 - 10.5*rstd^2)
	 *   d(x)[1] = rstd * (1 - 4*rstd * 3.5*rstd) = rstd * (1 - 14.0*rstd^2)
	 *   with rstd^2 = 1/12.5: d(x)[0] = rstd*0.16, d(x)[1] = rstd*(-0.12)
	 */
	param_clear();
	double in_d[] = {3.0, 4.0};
	double w_d[] = {1.0, 1.0};
	TensorHandle input = tensor_create_2d_f64(1, 2, hcopy(in_d, 2), 1);
	TensorHandle weight = tensor_create_1d_f64(2, hcopy(w_d, 2), 1);
	param_register("input", input);   /* param_idx 0 */
	param_register("weight", weight); /* param_idx 1 */
	double eps = 1e-9;
	TensorHandle loss = tensor_sum(tensor_rms_norm_2d(input, weight, eps));
	tensor_backward(loss);

	double rstd = 1.0 / sqrt(12.5 + eps);
	double r2 = rstd * rstd;
	double dx0 = rstd * (1.0 - 10.5 * r2);
	double dx1 = rstd * (1.0 - 14.0 * r2);
	cr_assert_float_eq(param_grad_item_at(0, 0), dx0, TEST_TOL_RELAXED,
	                   "d(x)[0] expected %.9f got %.9f", dx0, param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), dx1, TEST_TOL_RELAXED,
	                   "d(x)[1] expected %.9f got %.9f", dx1, param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(1, 0), 3.0 * rstd, TEST_TOL_RELAXED,
	                   "d(w)[0] expected %.9f got %.9f", 3.0 * rstd, param_grad_item_at(1, 0));
	cr_assert_float_eq(param_grad_item_at(1, 1), 4.0 * rstd, TEST_TOL_RELAXED,
	                   "d(w)[1] expected %.9f got %.9f", 4.0 * rstd, param_grad_item_at(1, 1));
}

Test(nn_norm_rms_norm, backward_weight_only) {
	/* Only weight requires grad; input does not. Exercises the
	 * weight-grad branch while the input-grad branch is skipped
	 * (a->requires_grad == 0).
	 *   Input [[1, 2]], mean_sq = 2.5, rstd = 1/sqrt(2.5)
	 *   x_hat = [1*rstd, 2*rstd]; dout = [1, 1]
	 *   d(weight)[j] = x_hat[j].
	 */
	param_clear();
	double in_d[] = {1.0, 2.0};
	double w_d[] = {1.0, 1.0};
	TensorHandle input = tensor_create_2d_f64(1, 2, hcopy(in_d, 2), 0);
	TensorHandle weight = tensor_create_1d_f64(2, hcopy(w_d, 2), 1);
	param_register("weight", weight); /* param_idx 0 */
	double eps = 1e-9;
	TensorHandle loss = tensor_sum(tensor_rms_norm_2d(input, weight, eps));
	tensor_backward(loss);

	double rstd = 1.0 / sqrt(2.5 + eps);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0 * rstd, TEST_TOL_RELAXED,
	                   "d(w)[0] expected %.9f got %.9f", 1.0 * rstd, param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 2.0 * rstd, TEST_TOL_RELAXED,
	                   "d(w)[1] expected %.9f got %.9f", 2.0 * rstd, param_grad_item_at(0, 1));
}

#ifdef BACKEND_TAPE
Test(nn_norm_rms_norm, backward_input_only_multirow) {
	/* Only input requires grad; two rows, each normalized independently.
	 * Validates the per-row d(x) loop with weight != 1 and the input-grad
	 * branch while the weight-grad branch is skipped.
	 *   weight = [2, 3]; eps -> 0; loss = sum(out); dout = ones.
	 * Per row, with dout_w = weight:
	 *   sum_k dout_w[k]*x_hat[k] ; mean = sum/n
	 *   d(x)[i,j] = rstd_i * (weight[j] - x_hat[i,j] * mean_i)
	 *
	 * Computed host-side and compared against the fused backward.
	 */
	param_clear();
	double in_d[] = {1.0, 2.0, 3.0, 4.0}; /* rows: [1,2], [3,4] */
	double w_d[] = {2.0, 3.0};
	TensorHandle input = tensor_create_2d_f64(2, 2, hcopy(in_d, 4), 1);
	TensorHandle weight = tensor_create_1d_f64(2, hcopy(w_d, 2), 0);
	param_register("input", input); /* param_idx 0 */
	double eps = 1e-9;
	TensorHandle loss = tensor_sum(tensor_rms_norm_2d(input, weight, eps));
	tensor_backward(loss);

	for (int i = 0; i < 2; i++) {
		double s = 0;
		for (int j = 0; j < 2; j++)
			s += in_d[i * 2 + j] * in_d[i * 2 + j];
		double rstd = 1.0 / sqrt(s / 2.0 + eps);
		double xhat[2];
		double mean = 0;
		for (int j = 0; j < 2; j++) {
			xhat[j] = in_d[i * 2 + j] * rstd;
			mean += w_d[j] * xhat[j];
		}
		mean /= 2.0;
		for (int j = 0; j < 2; j++) {
			double expect = rstd * (w_d[j] - xhat[j] * mean);
			cr_assert_float_eq(param_grad_item_at(0, i * 2 + j), expect, TEST_TOL_RELAXED,
			                   "d(x)[%d,%d] expected %.9f got %.9f", i, j, expect,
			                   param_grad_item_at(0, i * 2 + j));
		}
	}
}
#endif /* BACKEND_TAPE */

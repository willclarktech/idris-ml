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
#include "../../../../backend.h"
#include "test_helpers.h"

static double* heap_copy(const double* src, int n) {
    double* buf = (double*)malloc(n * sizeof(double));
    memcpy(buf, src, n * sizeof(double));
    return buf;
}

Test(nn_norm_rms_norm, forward_unit_weight) {
    /* Input: [[1, 2, 3, 4]], weight = [1, 1, 1, 1], eps = 1e-6.
     * mean_sq = (1 + 4 + 9 + 16) / 4 = 7.5
     * rstd    = 1 / sqrt(7.5 + 1e-6) ≈ 0.36514837...
     * out[j]  = input[j] * rstd
     */
    param_clear();
    double in_d[] = {1.0, 2.0, 3.0, 4.0};
    double w_d[]  = {1.0, 1.0, 1.0, 1.0};
    TensorHandle input  = tensor_create_2d_f64(1, 4, heap_copy(in_d, 4), 0);
    TensorHandle weight = tensor_create_1d_f64(4, heap_copy(w_d, 4), 0);
    double eps = 1e-6;
    TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
    double buf[4];
    tensor_to_doubles(r, buf);
    double rstd = 1.0 / sqrt(7.5 + eps);
    double expect[] = {1.0 * rstd, 2.0 * rstd, 3.0 * rstd, 4.0 * rstd};
    for (int j = 0; j < 4; j++) {
        cr_assert_float_eq(buf[j], expect[j], TEST_TOL_RELAXED,
            "rms_norm[%d] should be %.9f (got %.9f)",
            j, expect[j], buf[j]);
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
    double in_d[] = {1.0, 1.0, 1.0, 1.0,
                     2.0, 2.0, 2.0, 2.0};
    double w_d[]  = {1.0, 1.0, 1.0, 1.0};
    TensorHandle input  = tensor_create_2d_f64(2, 4, heap_copy(in_d, 8), 0);
    TensorHandle weight = tensor_create_1d_f64(4, heap_copy(w_d, 4), 0);
    double eps = 1e-6;
    TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
    double buf[8];
    tensor_to_doubles(r, buf);
    double rstd0 = 1.0 / sqrt(1.0 + eps);
    double rstd1 = 1.0 / sqrt(4.0 + eps);
    for (int j = 0; j < 4; j++) {
        cr_assert_float_eq(buf[j],     1.0 * rstd0, TEST_TOL_RELAXED,
            "row0[%d] expected %.9f got %.9f", j, 1.0 * rstd0, buf[j]);
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
    double w_d[]  = {2.0, 3.0};
    TensorHandle input  = tensor_create_2d_f64(1, 2, heap_copy(in_d, 2), 0);
    TensorHandle weight = tensor_create_1d_f64(2, heap_copy(w_d, 2), 0);
    double eps = 1e-6;
    TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
    double buf[2];
    tensor_to_doubles(r, buf);
    double rstd = 1.0 / sqrt(12.5 + eps);
    cr_assert_float_eq(buf[0], 6.0 * rstd, TEST_TOL_RELAXED,
        "weighted[0] expected %.9f got %.9f", 6.0 * rstd, buf[0]);
    cr_assert_float_eq(buf[1], 12.0 * rstd, TEST_TOL_RELAXED,
        "weighted[1] expected %.9f got %.9f", 12.0 * rstd, buf[1]);
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
    for (int i = 0; i < 32; i++) in_d[i] = (i % 5 == 0) ? -0.7 : 0.3 + (i * 0.11);
    for (int j = 0; j < 8;  j++) w_d[j]  = 0.5 + j * 0.1;
    TensorHandle input  = tensor_create_2d_f64(4, 8, heap_copy(in_d, 32), 0);
    TensorHandle weight = tensor_create_1d_f64(8, heap_copy(w_d, 8), 0);
    double eps = 1e-5;
    TensorHandle r = tensor_rms_norm_2d(input, weight, eps);
    double got[32];
    tensor_to_doubles(r, got);
    for (int i = 0; i < 4; i++) {
        double s = 0;
        for (int j = 0; j < 8; j++) s += in_d[i*8+j] * in_d[i*8+j];
        double rstd = 1.0 / sqrt(s / 8.0 + eps);
        for (int j = 0; j < 8; j++) {
            double expect = in_d[i*8+j] * rstd * w_d[j];
            cr_assert_float_eq(got[i*8+j], expect, TEST_TOL_RELAXED,
                "rms_norm[%d,%d] expected %.9f got %.9f", i, j, expect, got[i*8+j]);
        }
    }
}

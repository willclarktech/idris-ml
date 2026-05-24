/* Criterion suite for tensor_group_norm (forward).
 *
 *   group_norm(input, gamma, beta, numGroups, channels, spatial, eps):
 *     For each group of channels: compute mean + var over the group's
 *     spatial dimension, normalize, then scale by gamma + shift by beta
 *     (per-channel).
 *
 * Input shape: [N, C, S] = [N=1, C=numGroups*chansPerGroup, S=spatial].
 *
 * Closes the "tensor_group_norm 0 hits" probe gap on all three backends.
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

Test(nn_norm_group_norm, forward_one_group_unit_scale) {
    /* numGroups=1, channels=1, spatial=4 -> one group of 4 values.
     * input = [[[1, 2, 3, 4]]] (B=1, C=1, S=4)
     * gamma = [1], beta = [0], eps = 1e-5
     * mean = 2.5, var = 1.25, std = sqrt(1.25+eps) ≈ 1.1180...
     * output[i] = (input[i] - 2.5) / std
     *   = [-1.342, -0.447, 0.447, 1.342]
     */
    param_clear();
    double in_d[] = {1.0, 2.0, 3.0, 4.0};
    double g_d[] = {1.0};
    double b_d[] = {0.0};
    /* Construct flat 4-element tensor; group_norm doesn't care about
     * the rank label since channels*spatial = numel and the shape is
     * just used internally. tape's impl reads numel as channels*spatial. */
    TensorHandle input = tensor_create_2d_f64(1, 4, heap_copy(in_d, 4), 0);
    /* gamma/beta MUST be 1D length=channels per libtorch's group_norm
     * contract; tape's impl accepts any rank as flat. Use 1D for both. */
    TensorHandle gamma = tensor_create_1d_f64(1, heap_copy(g_d, 1), 0);
    TensorHandle beta  = tensor_create_1d_f64(1, heap_copy(b_d, 1), 0);
    double eps = 1e-5;
    TensorHandle r = tensor_group_norm(input, gamma, beta, 1, 1, 4, eps);
    double buf[4];
    tensor_to_doubles(r, buf);
    double std_val = sqrt(1.25 + eps);
    double expect[] = {(1.0 - 2.5) / std_val,
                       (2.0 - 2.5) / std_val,
                       (3.0 - 2.5) / std_val,
                       (4.0 - 2.5) / std_val};
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(buf[i], expect[i], TEST_TOL_RELAXED,
            "group_norm[%d] should be %.9f (got %.9f)",
            i, expect[i], buf[i]);
    }
}

Test(nn_norm_group_norm, output_zero_mean_within_group) {
    /* Stronger property: regardless of input values, the per-group output
     * should have mean ~0 (within F32 sum tol) when gamma=1, beta=0. */
    param_clear();
    double in_d[] = {5.0, -3.0, 7.0, -9.0, 1.0, 6.0};
    double g_d[] = {1.0};
    double b_d[] = {0.0};
    TensorHandle input = tensor_create_2d_f64(1, 6, heap_copy(in_d, 6), 0);
    /* gamma/beta MUST be 1D length=channels per libtorch's group_norm
     * contract; tape's impl accepts any rank as flat. Use 1D for both. */
    TensorHandle gamma = tensor_create_1d_f64(1, heap_copy(g_d, 1), 0);
    TensorHandle beta  = tensor_create_1d_f64(1, heap_copy(b_d, 1), 0);
    TensorHandle r = tensor_group_norm(input, gamma, beta, 1, 1, 6, 1e-5);
    double buf[6];
    tensor_to_doubles(r, buf);
    double sum = 0.0;
    for (int i = 0; i < 6; i++) sum += buf[i];
    cr_assert_float_eq(sum, 0.0, TEST_TOL_RELAXED,
        "group_norm output should be zero-mean (got sum %.9f)", sum);
}

/* Criterion suite for tensor_swiglu_2d (forward).
 *
 *   swiglu_2d(gate, up):
 *     out[i, j] = silu(gate[i, j]) * up[i, j]
 *              = gate[i, j] * sigmoid(gate[i, j]) * up[i, j]
 *
 * Replaces the tsilu + tmul pair in HfLlama.applyMlp with one fused
 * FFI call. Both inputs share shape [M, N].
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../../../../backend.h"
#include "../../test_helpers.h"

static double* heap_copy(const double* src, int n) {
    double* buf = (double*)malloc(n * sizeof(double));
    memcpy(buf, src, n * sizeof(double));
    return buf;
}

static double silu_ref(double x) {
    return x / (1.0 + exp(-x));
}

Test(nn_activation_swiglu, forward_zero_gate) {
    /* silu(0) = 0 * 0.5 = 0, so out should be 0 regardless of up. */
    param_clear();
    double g_d[] = {0.0, 0.0, 0.0, 0.0};
    double u_d[] = {1.0, 2.0, 3.0, 4.0};
    TensorHandle gate = tensor_create_2d_f64(1, 4, heap_copy(g_d, 4), 0);
    TensorHandle up   = tensor_create_2d_f64(1, 4, heap_copy(u_d, 4), 0);
    TensorHandle r = tensor_swiglu_2d(gate, up);
    double buf[4];
    tensor_to_doubles(r, buf);
    for (int j = 0; j < 4; j++) {
        cr_assert_float_eq(buf[j], 0.0, TEST_TOL_RELAXED,
            "swiglu[%d] should be 0 when gate=0 (got %.9f)", j, buf[j]);
    }
}

Test(nn_activation_swiglu, forward_unit_up) {
    /* up = 1, so out[j] = silu(gate[j]). Reduces to silu reference. */
    param_clear();
    double g_d[] = {1.0, -1.0, 2.0, -2.0};
    double u_d[] = {1.0, 1.0, 1.0, 1.0};
    TensorHandle gate = tensor_create_2d_f64(1, 4, heap_copy(g_d, 4), 0);
    TensorHandle up   = tensor_create_2d_f64(1, 4, heap_copy(u_d, 4), 0);
    TensorHandle r = tensor_swiglu_2d(gate, up);
    double buf[4];
    tensor_to_doubles(r, buf);
    for (int j = 0; j < 4; j++) {
        double expect = silu_ref(g_d[j]);
        cr_assert_float_eq(buf[j], expect, TEST_TOL_RELAXED,
            "swiglu[%d] expected %.9f got %.9f (gate=%.3f)",
            j, expect, buf[j], g_d[j]);
    }
}

Test(nn_activation_swiglu, forward_per_row_independent) {
    /* Two rows, distinct gate/up combinations.
     * row 0: gate=[1, -1], up=[2, 2] -> [2*silu(1), 2*silu(-1)]
     * row 1: gate=[0.5, 0.5], up=[1, -1] -> [silu(0.5), -silu(0.5)]
     */
    param_clear();
    double g_d[] = {1.0, -1.0, 0.5, 0.5};
    double u_d[] = {2.0, 2.0, 1.0, -1.0};
    TensorHandle gate = tensor_create_2d_f64(2, 2, heap_copy(g_d, 4), 0);
    TensorHandle up   = tensor_create_2d_f64(2, 2, heap_copy(u_d, 4), 0);
    TensorHandle r = tensor_swiglu_2d(gate, up);
    double buf[4];
    tensor_to_doubles(r, buf);
    for (int k = 0; k < 4; k++) {
        double expect = silu_ref(g_d[k]) * u_d[k];
        cr_assert_float_eq(buf[k], expect, TEST_TOL_RELAXED,
            "swiglu[%d] expected %.9f got %.9f (gate=%.3f up=%.3f)",
            k, expect, buf[k], g_d[k], u_d[k]);
    }
}

Test(nn_activation_swiglu, forward_matches_decomposed_chain) {
    /* Strongest correctness check: fused op must match host-side
     * silu(g) * u over a non-trivial [seq, intermediate] grid. Same
     * shape class HfLlama's MLP hits per token at miniature scale.
     */
    param_clear();
    double g_d[32];
    double u_d[32];
    for (int i = 0; i < 32; i++) {
        g_d[i] = (i % 5 == 0) ? -0.7 : 0.3 + (i * 0.11);
        u_d[i] = 0.5 - (i * 0.07);
    }
    TensorHandle gate = tensor_create_2d_f64(4, 8, heap_copy(g_d, 32), 0);
    TensorHandle up   = tensor_create_2d_f64(4, 8, heap_copy(u_d, 32), 0);
    TensorHandle r = tensor_swiglu_2d(gate, up);
    double got[32];
    tensor_to_doubles(r, got);
    for (int k = 0; k < 32; k++) {
        double expect = silu_ref(g_d[k]) * u_d[k];
        cr_assert_float_eq(got[k], expect, TEST_TOL_RELAXED,
            "swiglu[%d] expected %.9f got %.9f", k, expect, got[k]);
    }
}

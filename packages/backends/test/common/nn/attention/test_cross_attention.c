/* Criterion suite for tensor_cross_attention (forward).
 *
 *   cross_attention(Q, K, V, mask, scale) = softmax(Q @ K^T * scale) @ V
 *   shapes: Q [B, seqQ, d], K [B, seqK, d], V [B, seqK, d] -> [B, seqQ, d]
 *
 * Tape composes from transpose_last2 + bmm_3x3 + softmax_3d + (optional)
 * masked_fill — those are all individually tested elsewhere. This test
 * verifies the composition produces the right per-cell value.
 *
 * On torch the path is hand-composed; on mlx too. So this test
 * covers all three backends' implementations of the same algorithm.
 *
 * Closes part of W3b (cross_attention is in the torch custom-logic
 * list) and also closes the "tensor_cross_attention 0 hits" probe gap.
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

Test(nn_attention_cross_attention, forward_no_mask) {
    /* Q = [[[1, 0]]]                    shape [1, 1, 2]
     * K = [[[1, 0], [0, 1]]]            shape [1, 2, 2]
     * V = [[[10, 20], [30, 40]]]        shape [1, 2, 2]
     *
     * KT             = [[[1, 0], [0, 1]]]
     * scores = Q@KT  = [[[1, 0]]]
     * scale = 1.0    -> scores = [[[1, 0]]]
     * softmax row 0: [e/(e+1), 1/(e+1)] = [0.731058578630, 0.268941421370]
     * result = attn @ V = [[[0.7311*10 + 0.2689*30, 0.7311*20 + 0.2689*40]]]
     *                   = [[[15.37788..., 25.37788...]]]
     */
    param_clear();
    double Qd[] = {1.0, 0.0};
    double Kd[] = {1.0, 0.0, 0.0, 1.0};
    double Vd[] = {10.0, 20.0, 30.0, 40.0};
    TensorHandle Qflat = tensor_create_2d_f64(1, 2, heap_copy(Qd, 2), 0);
    TensorHandle Kflat = tensor_create_2d_f64(2, 2, heap_copy(Kd, 4), 0);
    TensorHandle Vflat = tensor_create_2d_f64(2, 2, heap_copy(Vd, 4), 0);
    TensorHandle Q = tensor_reshape_3d(Qflat, 1, 1, 2);
    TensorHandle K = tensor_reshape_3d(Kflat, 1, 2, 2);
    TensorHandle V = tensor_reshape_3d(Vflat, 1, 2, 2);

    TensorHandle r = tensor_cross_attention(Q, K, V, (TensorHandle)0, 1.0);

    double buf[2];
    tensor_to_doubles(r, buf);

    double e = exp(1.0);
    double w0 = e / (e + 1.0);
    double w1 = 1.0 / (e + 1.0);
    double expect_0 = w0 * 10.0 + w1 * 30.0;
    double expect_1 = w0 * 20.0 + w1 * 40.0;
    cr_assert_float_eq(buf[0], expect_0, TEST_TOL_RELAXED,
        "cross_attn[0,0,0] should be %.9f (got %.9f)", expect_0, buf[0]);
    cr_assert_float_eq(buf[1], expect_1, TEST_TOL_RELAXED,
        "cross_attn[0,0,1] should be %.9f (got %.9f)", expect_1, buf[1]);
}

Test(nn_attention_cross_attention, forward_with_mask) {
    /* Same Q, K, V but mask = [[[0, 1]]] (mask out the 2nd key position).
     * masked_fill(scores, mask, -1e20) zeroes out position 1 of the softmax,
     * so attention collapses onto position 0 and result = V[0] = [10, 20].
     */
    param_clear();
    double Qd[] = {1.0, 0.0};
    double Kd[] = {1.0, 0.0, 0.0, 1.0};
    double Vd[] = {10.0, 20.0, 30.0, 40.0};
    double Md[] = {0.0, 1.0};
    TensorHandle Qflat = tensor_create_2d_f64(1, 2, heap_copy(Qd, 2), 0);
    TensorHandle Kflat = tensor_create_2d_f64(2, 2, heap_copy(Kd, 4), 0);
    TensorHandle Vflat = tensor_create_2d_f64(2, 2, heap_copy(Vd, 4), 0);
    TensorHandle Mflat = tensor_create_2d_f64(1, 2, heap_copy(Md, 2), 0);
    TensorHandle Q = tensor_reshape_3d(Qflat, 1, 1, 2);
    TensorHandle K = tensor_reshape_3d(Kflat, 1, 2, 2);
    TensorHandle V = tensor_reshape_3d(Vflat, 1, 2, 2);
    TensorHandle M = tensor_reshape_3d(Mflat, 1, 1, 2);

    TensorHandle r = tensor_cross_attention(Q, K, V, M, 1.0);

    double buf[2];
    tensor_to_doubles(r, buf);

    cr_assert_float_eq(buf[0], 10.0, TEST_TOL_RELAXED,
        "cross_attn[0,0,0] masked should be V[0,0] = 10 (got %.9f)", buf[0]);
    cr_assert_float_eq(buf[1], 20.0, TEST_TOL_RELAXED,
        "cross_attn[0,0,1] masked should be V[0,1] = 20 (got %.9f)", buf[1]);
}

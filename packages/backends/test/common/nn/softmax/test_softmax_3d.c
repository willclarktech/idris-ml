/* Criterion suite for tensor_softmax_3d (forward + backward).
 *
 *   softmax along last dim of [B, m, n]:
 *     y[b, i, j] = exp(x[b, i, j]) / sum_k exp(x[b, i, k])
 *
 * Closes the W3 OP_SOFTMAX_3D coverage gap on tape (mlx and torch both
 * already implement tensor_softmax_3d FFI — this test exercises all three
 * via the common path).
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
#include "../../../../backend.h"
#include "../../test_helpers.h"

static double* heap_copy(const double* src, int n) {
    double* buf = (double*)malloc(n * sizeof(double));
    memcpy(buf, src, n * sizeof(double));
    return buf;
}

Test(nn_softmax_softmax_3d, forward_two_class) {
    /* x = [[[1, 2]]] shape [1,1,2] -> y = [[[1/(1+e), e/(1+e)]]]. */
    param_clear();
    double xd[] = {1.0, 2.0};
    TensorHandle flat = tensor_create_param_2d_f64(1, 2, heap_copy(xd, 2));
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
    TensorHandle flat = tensor_create_param_2d_f64(1, 3, heap_copy(xd, 3));
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
    TensorHandle flat = tensor_create_param_2d_f64(1, 2, heap_copy(xd, 2));
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
        "grad x[0] for sum-loss should be 0 (got %.9f)",
        param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, TEST_TOL_RELAXED,
        "grad x[1] for sum-loss should be 0 (got %.9f)",
        param_grad_item_at(0, 1));
}

/* Criterion suite for tensor_cosine_similarity (forward + backward).
 *
 * Row-wise cosine: a [n,w] vs b [1,w] -> out [n], with
 *   cos(i) = (a[i] · b) / (||a[i]|| * ||b||)
 * tape/mlx add 1e-8 to each norm; torch uses libtorch's default 1e-8.
 *
 * Closes the W3/W4 OP_COSINE_SIM coverage gap on tape + mlx.
 * Surfaced by scripts/coverage-gap-probe.sh before this commit.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../../../../backend.h"
#include "test_helpers.h"

/* tape and mlx both add 1e-8 to each norm to dodge div-by-zero; libtorch
 * uses its default eps of 1e-8 too. That bias propagates to ~1e-8 in
 * the output on unit-norm inputs — well above both backends' TIGHT
 * tolerances. 1e-6 leaves room for that bias plus F32-summation
 * roundoff on mlx without admitting genuine numerical bugs. */
#define COSINE_TOL 1e-6

static double* heap_copy(const double* src, int n) {
    double* buf = (double*)malloc(n * sizeof(double));
    memcpy(buf, src, n * sizeof(double));
    return buf;
}

Test(nn_attention_cosine_similarity, forward_parallel_rows) {
    /* a = [[1, 0], [0, 1]], b = [[1, 0]] -> cos = [1, 0].
     * With the 1e-8 bias on both norms the "parallel" case lands within
     * COSINE_TOL of 1.0; the "perpendicular" case is exact 0. */
    param_clear();
    double ad[] = {1.0, 0.0, 0.0, 1.0};
    double bd[] = {1.0, 0.0};
    TensorHandle a = tensor_create_param_2d_f64(2, 2, heap_copy(ad, 4));
    TensorHandle b = tensor_create_param_2d_f64(1, 2, heap_copy(bd, 2));
    param_register("a", a);
    param_register("b", b);
    TensorHandle r = tensor_cosine_similarity(a, b, 1);
    cr_assert_float_eq(tensor_item_1d(r, 0), 1.0, COSINE_TOL,
        "cos(parallel) should be 1 (got %.9f)", tensor_item_1d(r, 0));
    cr_assert_float_eq(tensor_item_1d(r, 1), 0.0, COSINE_TOL,
        "cos(perpendicular) should be 0 (got %.9f)", tensor_item_1d(r, 1));
}

Test(nn_attention_cosine_similarity, forward_backward_a_grad) {
    /* a = [[3, 4]], b = [[1, 0]] -> cos = 3/(5*1) = 0.6.
     * Gradient of cos w.r.t. a[j] (no bias terms, exact):
     *   dcos/da[0] = b[0] / (||a|| ||b||) - cos * a[0] / ||a||^2
     *             = 1/5 - 0.6 * 3/25 = 0.2 - 0.072 = 0.128
     *   dcos/da[1] = 0/5 - 0.6 * 4/25 = -0.096
     * Loss = cos[0] so dL/dcos[0] = 1; the 1e-8 norm bias and 1e-10
     * squared-norm bias in the tape kernel both contribute << 1e-6
     * (within COSINE_TOL).
     */
    param_clear();
    double ad[] = {3.0, 4.0};
    double bd[] = {1.0, 0.0};
    TensorHandle a = tensor_create_param_2d_f64(1, 2, heap_copy(ad, 2));
    TensorHandle b = tensor_create_param_2d_f64(1, 2, heap_copy(bd, 2));
    param_register("a", a);
    param_register("b", b);

    TensorHandle r = tensor_cosine_similarity(a, b, 1);
    cr_assert_float_eq(tensor_item_1d(r, 0), 0.6, COSINE_TOL,
        "cos([3,4],[1,0]) should be 0.6 (got %.9f)", tensor_item_1d(r, 0));

    /* Backward: feed an explicit upstream grad of 1.0 on cos[0] by
     * using a sum reduction loss (= cos[0] since n=1). */
    TensorHandle loss = tensor_sum(r);
    tensor_backward(loss);

    cr_assert_float_eq(param_grad_item_at(0, 0), 0.128, COSINE_TOL,
        "grad a[0] should be 0.128 (got %.9f)", param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(0, 1), -0.096, COSINE_TOL,
        "grad a[1] should be -0.096 (got %.9f)", param_grad_item_at(0, 1));
}

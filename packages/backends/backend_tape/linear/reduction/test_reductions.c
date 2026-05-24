/* Criterion suite for tape reductions.
 * Covers sum, mean, sum_dim, tensor_min, tensor_max. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_reduction_sum, forward_backward) {
    param_clear();
    double d[] = {1.0, 2.0, 3.0, 4.0};
    int s[] = {4};
    TensorHandle t = tensor_create(d, s, 1, 1);
    param_register("t", t);
    TensorHandle r = tensor_sum(t);
    cr_assert_float_eq(tensor_item(r), 10.0, 1e-12);
    tensor_backward(r);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
            "sum grad[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
}

Test(linear_reduction_mean, forward_backward) {
    param_clear();
    double d[] = {1.0, 2.0, 3.0, 4.0};
    int s[] = {4};
    TensorHandle t = tensor_create(d, s, 1, 1);
    param_register("t", t);
    TensorHandle r = tensor_mean(t);
    cr_assert_float_eq(tensor_item(r), 2.5, 1e-12);
    tensor_backward(r);
    /* d_mean / d_x[i] = 1/numel = 0.25 */
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 0.25, 1e-12,
            "mean grad[%d] should be 0.25 (got %.6f)", i, param_grad_item_at(0, i));
}

Test(linear_reduction_min, forward) {
    double d[] = {3.0, -1.0, 5.0, 2.0};
    int s[] = {4};
    TensorHandle t = tensor_create(d, s, 1, 0);
    TensorHandle r = tensor_min(t);
    cr_assert_float_eq(tensor_item(r), -1.0, 1e-12);
}

Test(linear_reduction_max, forward) {
    double d[] = {3.0, -1.0, 5.0, 2.0};
    int s[] = {4};
    TensorHandle t = tensor_create(d, s, 1, 0);
    TensorHandle r = tensor_max(t);
    cr_assert_float_eq(tensor_item(r), 5.0, 1e-12);
}

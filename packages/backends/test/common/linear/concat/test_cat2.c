/* Criterion suite for tape `tensor_cat2`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(linear_concat_cat2, forward_two_vectors) {
    double ad[] = {1.0, 2.0, 3.0};
    double bd[] = {4.0, 5.0};
    int sa[] = {3};
    int sb[] = {2};
    TensorHandle a = tensor_create(ad, sa, 1, 0);
    TensorHandle b = tensor_create(bd, sb, 1, 0);
    TensorHandle c = tensor_cat2(a, b);
    cr_assert_eq(tensor_numel(c), 5);
    double out[5];
    tensor_to_doubles(c, out);
    cr_assert_float_eq(out[0], 1.0, 1e-12);
    cr_assert_float_eq(out[1], 2.0, 1e-12);
    cr_assert_float_eq(out[2], 3.0, 1e-12);
    cr_assert_float_eq(out[3], 4.0, 1e-12);
    cr_assert_float_eq(out[4], 5.0, 1e-12);
}

Test(linear_concat_cat2, backward_splits_grad) {
    /* c = cat2(a, b); loss = sum(c). d_a[i] = 1, d_b[j] = 1. */
    param_clear();
    double ad[] = {1.0, 2.0, 3.0};
    double bd[] = {4.0, 5.0};
    int sa[] = {3};
    int sb[] = {2};
    TensorHandle a = tensor_create(ad, sa, 1, 1);
    TensorHandle b = tensor_create(bd, sb, 1, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_cat2(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);
    for (int i = 0; i < 3; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
            "a's grad[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
    for (int j = 0; j < 2; j++)
        cr_assert_float_eq(param_grad_item_at(1, j), 1.0, 1e-12,
            "b's grad[%d] should be 1 (got %.6f)", j, param_grad_item_at(1, j));
}

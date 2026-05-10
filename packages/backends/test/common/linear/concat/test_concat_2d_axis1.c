/* Criterion suite for tape `tensor_concat_2d_axis1`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(linear_concat_concat_2d_axis1, forward_2x2_with_2x3) {
    /* A = [[1, 2], [3, 4]],  B = [[10, 20, 30], [40, 50, 60]]
       R = [[1, 2, 10, 20, 30], [3, 4, 40, 50, 60]] */
    double ad[] = {1.0, 2.0, 3.0, 4.0};
    double bd[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    int sa[] = {2, 2};
    int sb[] = {2, 3};
    TensorHandle A = tensor_create(ad, sa, 2, 0);
    TensorHandle B = tensor_create(bd, sb, 2, 0);
    TensorHandle R = tensor_concat_2d_axis1(A, B);
    cr_assert_eq(tensor_dim(R), 2);
    cr_assert_eq(tensor_size(R, 0), 2);
    cr_assert_eq(tensor_size(R, 1), 5);
    double out[10];
    tensor_to_doubles(R, out);
    double expected[] = {1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0};
    for (int i = 0; i < 10; i++) cr_assert_float_eq(out[i], expected[i], 1e-12);
}

Test(linear_concat_concat_2d_axis1, backward_splits_columnwise) {
    /* sum(concat(A, B)) -> d_A is all 1s shape (2,2); d_B is all 1s shape (2,3). */
    param_clear();
    double ad[] = {1.0, 2.0, 3.0, 4.0};
    double bd[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    int sa[] = {2, 2};
    int sb[] = {2, 3};
    TensorHandle A = tensor_create(ad, sa, 2, 1);
    TensorHandle B = tensor_create(bd, sb, 2, 1);
    param_register("A", A);
    param_register("B", B);
    TensorHandle R = tensor_concat_2d_axis1(A, B);
    TensorHandle loss = tensor_sum(R);
    tensor_backward(loss);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
            "A's grad[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
    for (int j = 0; j < 6; j++)
        cr_assert_float_eq(param_grad_item_at(1, j), 1.0, 1e-12,
            "B's grad[%d] should be 1 (got %.6f)", j, param_grad_item_at(1, j));
}

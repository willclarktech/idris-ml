/* Criterion suite for tape `tensor_narrow`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(tape_linear_shape_narrow, forward_slice) {
    double d[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    int s[] = {5};
    TensorHandle v = tensor_create(d, s, 1, 0);
    TensorHandle n = tensor_narrow(v, 0, 1, 3);  /* [20, 30, 40] */
    double out[3];
    tensor_to_doubles(n, out);
    cr_assert_float_eq(out[0], 20.0, 1e-12);
    cr_assert_float_eq(out[1], 30.0, 1e-12);
    cr_assert_float_eq(out[2], 40.0, 1e-12);
}

Test(tape_linear_shape_narrow, backward_scatters_to_offset) {
    /* narrow [v0..v4], 1..4 -> [v1, v2, v3]; sum -> backward should
       set parent's grad to [0, 1, 1, 1, 0]. */
    param_clear();
    double d[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    int s[] = {5};
    TensorHandle v = tensor_create(d, s, 1, 1);
    param_register("v", v);
    TensorHandle n = tensor_narrow(v, 0, 1, 3);
    TensorHandle loss = tensor_sum(n);
    tensor_backward(loss);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "grad[0] should be 0");
    cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12,
        "grad[1] should be 1.0 (got %.6f)", param_grad_item_at(0, 1));
    cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "grad[2] should be 1.0");
    cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12, "grad[3] should be 1.0");
    cr_assert_float_eq(param_grad_item_at(0, 4), 0.0, 1e-12, "grad[4] should be 0");
}

/* Criterion suite for tape `tensor_reshape`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(linear_shape_reshape, forward_2x3_to_3x2) {
    double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int s_in[]  = {2, 3};
    int s_out[] = {3, 2};
    TensorHandle t = tensor_create(d, s_in, 2, 0);
    TensorHandle r = tensor_reshape(t, s_out, 2);
    cr_assert_eq(tensor_dim(r), 2);
    cr_assert_eq(tensor_size(r, 0), 3);
    cr_assert_eq(tensor_size(r, 1), 2);
    cr_assert_eq(tensor_numel(r), 6);
    /* Data is shared, so the same elements show up flat. */
    double out[6];
    tensor_to_doubles(r, out);
    for (int i = 0; i < 6; i++) cr_assert_float_eq(out[i], d[i], 1e-12);
}

Test(linear_shape_reshape, backward_passthrough) {
    /* Forward reshape + sum to scalar — grad should be 1.0 at every position. */
    param_clear();
    double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int s_in[]  = {2, 3};
    int s_out[] = {3, 2};
    TensorHandle t = tensor_create(d, s_in, 2, 1);
    param_register("t", t);
    TensorHandle r = tensor_reshape(t, s_out, 2);
    TensorHandle loss = tensor_sum(r);
    tensor_backward(loss);
    for (int i = 0; i < 6; i++) {
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
            "reshape grad-passthrough: grad[%d] should be 1.0 (got %.6f)",
            i, param_grad_item_at(0, i));
    }
}

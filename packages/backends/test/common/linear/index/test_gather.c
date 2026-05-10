/* Criterion suite for tape `tensor_gather`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(linear_index_gather, forward_with_index) {
    /* input = [10, 20, 30, 40], index = [3, 1, 0] -> [40, 20, 10] */
    double id[] = {10.0, 20.0, 30.0, 40.0};
    double ixd[] = {3.0, 1.0, 0.0};
    int s_in[] = {4};
    int s_ix[] = {3};
    TensorHandle input = tensor_create(id, s_in, 1, 0);
    TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
    TensorHandle r = tensor_gather(input, index, 3);
    double out[3];
    tensor_to_doubles(r, out);
    cr_assert_float_eq(out[0], 40.0, 1e-12);
    cr_assert_float_eq(out[1], 20.0, 1e-12);
    cr_assert_float_eq(out[2], 10.0, 1e-12);
}

Test(linear_index_gather, backward_scatters_grad) {
    /* gather with index [3, 1, 0]; sum -> d_input scattered to [3,1,0]. */
    param_clear();
    double id[] = {10.0, 20.0, 30.0, 40.0};
    double ixd[] = {3.0, 1.0, 0.0};
    int s_in[] = {4};
    int s_ix[] = {3};
    TensorHandle input = tensor_create(id, s_in, 1, 1);
    TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
    param_register("input", input);
    TensorHandle r = tensor_gather(input, index, 3);
    TensorHandle loss = tensor_sum(r);
    tensor_backward(loss);
    /* d_input should be [1, 1, 0, 1] — positions 0, 1, 3 picked, 2 unpicked */
    cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12,
        "input grad[1] should be 1 (got %.6f)", param_grad_item_at(0, 1));
    cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12);
}

/* Criterion suite for tape `tensor_select`. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_shape_select, forward_vector_element) {
    double d[] = {10.0, 20.0, 30.0, 40.0};
    int s[] = {4};
    TensorHandle v = tensor_create(d, s, 1, 0);
    TensorHandle e = tensor_select(v, 0, 2);
    cr_assert_float_eq(tensor_item(e), 30.0, 1e-12);
}

Test(linear_shape_select, backward_scatters_to_index) {
    /* Vector [a0, a1, a2, a3]; select index 1; backward should put 1.0
       at a's grad[1] and 0.0 elsewhere. */
    param_clear();
    double d[] = {10.0, 20.0, 30.0, 40.0};
    int s[] = {4};
    TensorHandle v = tensor_create(d, s, 1, 1);
    param_register("v", v);
    TensorHandle picked = tensor_select(v, 0, 1);
    tensor_backward(picked);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "grad[0] should be 0");
    cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12,
        "grad[1] should be 1.0 (got %.6f)", param_grad_item_at(0, 1));
    cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12, "grad[2] should be 0");
    cr_assert_float_eq(param_grad_item_at(0, 3), 0.0, 1e-12, "grad[3] should be 0");
}

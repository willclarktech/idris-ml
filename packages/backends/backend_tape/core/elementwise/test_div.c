/* Criterion suite for tape `tensor_div`. */

#include <criterion/criterion.h>
#include "backend.h"

Test(core_elementwise_div, forward_scalar) {
    TensorHandle a = tensor_create_scalar(10.0, 0);
    TensorHandle b = tensor_create_scalar(4.0, 0);
    TensorHandle c = tensor_div(a, b);
    cr_assert_float_eq(tensor_item(c), 2.5, 1e-12);
}

Test(core_elementwise_div, backward_scalar) {
    /* c = a/b; dc/da = 1/b, dc/db = -a/b^2 */
    param_clear();
    TensorHandle a = tensor_create_scalar(10.0, 1);
    TensorHandle b = tensor_create_scalar(4.0, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_div(a, b);
    tensor_backward(c);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, 1e-12,
        "d(a/b)/da should be 1/b=0.25 (got %.6f)", param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(1, 0), -0.625, 1e-12,
        "d(a/b)/db should be -a/b^2=-0.625 (got %.6f)", param_grad_item_at(1, 0));
}

Test(core_elementwise_div, backward_vector) {
    param_clear();
    double ad[] = {6.0, 12.0};
    double bd[] = {2.0, 4.0};
    int s[] = {2};
    TensorHandle a = tensor_create(ad, s, 1, 1);
    TensorHandle b = tensor_create(bd, s, 1, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_div(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);
    /* d(sum(a/b))/da[i] = 1/b[i], d(sum(a/b))/db[i] = -a[i]/b[i]^2 */
    cr_assert_float_eq(param_grad_item_at(0, 0), 1.0/2.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(0, 1), 1.0/4.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(1, 0), -6.0/(2.0*2.0), 1e-12);
    cr_assert_float_eq(param_grad_item_at(1, 1), -12.0/(4.0*4.0), 1e-12);
}

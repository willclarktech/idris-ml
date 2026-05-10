/* Criterion suite for tape `tensor_sub`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(tape_core_elementwise_sub, forward_scalar) {
    TensorHandle a = tensor_create_scalar(10.0, 0);
    TensorHandle b = tensor_create_scalar(3.0, 0);
    TensorHandle c = tensor_sub(a, b);
    cr_assert_float_eq(tensor_item(c), 7.0, 1e-12);
}

Test(tape_core_elementwise_sub, forward_vector) {
    double ad[] = {10.0, 20.0, 30.0};
    double bd[] = {1.0, 2.0, 3.0};
    int s[] = {3};
    TensorHandle a = tensor_create(ad, s, 1, 0);
    TensorHandle b = tensor_create(bd, s, 1, 0);
    TensorHandle c = tensor_sub(a, b);
    double out[3];
    tensor_to_doubles(c, out);
    cr_assert_float_eq(out[0],  9.0, 1e-12);
    cr_assert_float_eq(out[1], 18.0, 1e-12);
    cr_assert_float_eq(out[2], 27.0, 1e-12);
}

Test(tape_core_elementwise_sub, backward_scalar_grads_signs) {
    /* c = a - b; dc/da = +1, dc/db = -1 */
    param_clear();
    TensorHandle a = tensor_create_scalar(10.0, 1);
    TensorHandle b = tensor_create_scalar(3.0, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_sub(a, b);
    tensor_backward(c);
    cr_assert_float_eq(param_grad_item_at(0, 0),  1.0, 1e-12,
        "d(a-b)/da should be +1.0 (got %.6f)", param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(1, 0), -1.0, 1e-12,
        "d(a-b)/db should be -1.0 (got %.6f)", param_grad_item_at(1, 0));
}

Test(tape_core_elementwise_sub, backward_vector_signs) {
    param_clear();
    double ad[] = {5.0, 6.0, 7.0};
    double bd[] = {1.0, 2.0, 3.0};
    int s[] = {3};
    TensorHandle a = tensor_create(ad, s, 1, 1);
    TensorHandle b = tensor_create(bd, s, 1, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_sub(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);
    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(param_grad_item_at(0, i),  1.0, 1e-12,
            "d(sum(a-b))/da[%d] should be +1.0", i);
        cr_assert_float_eq(param_grad_item_at(1, i), -1.0, 1e-12,
            "d(sum(a-b))/db[%d] should be -1.0", i);
    }
}

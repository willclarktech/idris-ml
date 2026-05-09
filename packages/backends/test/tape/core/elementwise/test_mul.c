/* Criterion suite for tape `tensor_mul` (Phase 1a.4). */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(tape_core_elementwise_mul, forward_scalar) {
    TensorHandle a = tensor_create_scalar(3.0, 0);
    TensorHandle b = tensor_create_scalar(4.0, 0);
    TensorHandle c = tensor_mul(a, b);
    cr_assert_float_eq(tensor_item(c), 12.0, 1e-12);
}

Test(tape_core_elementwise_mul, forward_vector) {
    double ad[] = {1.0, 2.0, 3.0};
    double bd[] = {10.0, 20.0, 30.0};
    int s[] = {3};
    TensorHandle a = tensor_create(ad, s, 1, 0);
    TensorHandle b = tensor_create(bd, s, 1, 0);
    TensorHandle c = tensor_mul(a, b);
    double out[3];
    tensor_to_doubles(c, out);
    cr_assert_float_eq(out[0], 10.0, 1e-12);
    cr_assert_float_eq(out[1], 40.0, 1e-12);
    cr_assert_float_eq(out[2], 90.0, 1e-12);
}

Test(tape_core_elementwise_mul, backward_scalar_swapped_grads) {
    /* c = a*b; dc/da = b, dc/db = a */
    param_clear();
    TensorHandle a = tensor_create_scalar(3.0, 1);
    TensorHandle b = tensor_create_scalar(4.0, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_mul(a, b);
    tensor_backward(c);
    cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, 1e-12,
        "d(a*b)/da should be b=4.0 (got %.6f)", param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(1, 0), 3.0, 1e-12,
        "d(a*b)/db should be a=3.0 (got %.6f)", param_grad_item_at(1, 0));
}

Test(tape_core_elementwise_mul, backward_vector_swap) {
    param_clear();
    double ad[] = {2.0, 3.0, 5.0};
    double bd[] = {7.0, 11.0, 13.0};
    int s[] = {3};
    TensorHandle a = tensor_create(ad, s, 1, 1);
    TensorHandle b = tensor_create(bd, s, 1, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_mul(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);
    /* d(sum(a*b))/da[i] = b[i], d(sum(a*b))/db[i] = a[i] */
    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(param_grad_item_at(0, i), bd[i], 1e-12,
            "d_a[%d] should be b[%d]=%.1f", i, i, bd[i]);
        cr_assert_float_eq(param_grad_item_at(1, i), ad[i], 1e-12,
            "d_b[%d] should be a[%d]=%.1f", i, i, ad[i]);
    }
}

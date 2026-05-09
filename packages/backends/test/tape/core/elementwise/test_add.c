/* Criterion suite for tape `tensor_add` (Phase 1a.2).
 *
 * First per-op extraction with a real backward — exercises both
 * forward correctness and the dispatch-table-driven backward
 * (registered via TAPE_REGISTER_OP(OP_ADD, tape_backward_add) at
 * file scope of core/elementwise/add.c).
 *
 * The "RED before this commit" assertion is grad_a[0] == 1.0 after
 * c = a + b; backward(c). With the monolith's `case OP_ADD:` deleted
 * and TAPE_REGISTER_OP not yet added, backward falls through the
 * dispatch table (returns NULL) AND the switch (no case) — grad_a
 * stays at the initialized 0.0. Test fails: got 0.0, expected 1.0.
 * Registering the backward via TAPE_REGISTER_OP flips it green.
 */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(tape_core_elementwise_add, forward_scalar_scalar) {
    TensorHandle a = tensor_create_scalar(3.0, 1);
    TensorHandle b = tensor_create_scalar(4.0, 1);
    TensorHandle c = tensor_add(a, b);
    cr_assert_float_eq(tensor_item(c), 7.0, 1e-12);
}

Test(tape_core_elementwise_add, forward_vector_vector_same_shape) {
    double ad[] = {1.0, 2.0, 3.0};
    double bd[] = {10.0, 20.0, 30.0};
    int s[] = {3};
    TensorHandle a = tensor_create(ad, s, 1, 0);
    TensorHandle b = tensor_create(bd, s, 1, 0);
    TensorHandle c = tensor_add(a, b);
    double out[3];
    tensor_to_doubles(c, out);
    cr_assert_float_eq(out[0], 11.0, 1e-12);
    cr_assert_float_eq(out[1], 22.0, 1e-12);
    cr_assert_float_eq(out[2], 33.0, 1e-12);
}

Test(tape_core_elementwise_add, backward_scalar_grads_both_one) {
    /* c = a + b; dc/da = dc/db = 1 (sum) */
    param_clear();
    TensorHandle a = tensor_create_scalar(3.0, 1);
    TensorHandle b = tensor_create_scalar(4.0, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_add(a, b);
    tensor_backward(c);
    cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12,
        "d(a+b)/da should be 1.0 (got %.6f)", param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(1, 0), 1.0, 1e-12,
        "d(a+b)/db should be 1.0 (got %.6f)", param_grad_item_at(1, 0));
}

Test(tape_core_elementwise_add, backward_vector_vector_same_shape) {
    /* c = a + b; reduce to scalar via sum; dscalar/d_each_input = 1 */
    param_clear();
    double ad[] = {1.0, 2.0, 3.0};
    double bd[] = {10.0, 20.0, 30.0};
    int s[] = {3};
    TensorHandle a = tensor_create(ad, s, 1, 1);
    TensorHandle b = tensor_create(bd, s, 1, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_add(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);
    /* Each element of a's grad should be 1.0 */
    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
            "d(sum(a+b))/da[%d] should be 1.0 (got %.6f)", i, param_grad_item_at(0, i));
    }
    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(param_grad_item_at(1, i), 1.0, 1e-12,
            "d(sum(a+b))/db[%d] should be 1.0 (got %.6f)", i, param_grad_item_at(1, i));
    }
}

Test(tape_core_elementwise_add, backward_scalar_plus_vector_broadcast) {
    /* c = scalar + vector; d_scalar = sum(d_c), d_vector[i] = d_c[i].
       After loss = sum(c), d_c[i] = 1.0, so d_scalar = numel(c), d_vector[i] = 1. */
    param_clear();
    TensorHandle a = tensor_create_scalar(5.0, 1);
    double bd[] = {1.0, 2.0, 3.0, 4.0};
    int s[] = {4};
    TensorHandle b = tensor_create(bd, s, 1, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_add(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);
    cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, 1e-12,
        "d(sum(scalar+vec))/d_scalar should be 4.0 (got %.6f)", param_grad_item_at(0, 0));
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(param_grad_item_at(1, i), 1.0, 1e-12,
            "d(sum(scalar+vec))/d_vec[%d] should be 1.0", i);
    }
}

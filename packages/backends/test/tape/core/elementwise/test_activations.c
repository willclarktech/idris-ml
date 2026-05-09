/* Criterion suite for tape elementwise activations (Phase 1a.8).
 * Covers sigmoid, tanh, softplus. */

#include <criterion/criterion.h>
#include <math.h>
#include "../../../../backend.h"

Test(tape_core_elementwise_sigmoid, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(0.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_sigmoid(a);
    cr_assert_float_eq(tensor_item(r), 0.5, 1e-12);
    tensor_backward(r);
    /* d sigmoid(x)/dx at x=0 = 0.5 * (1 - 0.5) = 0.25 */
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, 1e-12,
        "d sigmoid(0)/dx should be 0.25 (got %.6f)", param_grad_item_at(0, 0));
}

Test(tape_core_elementwise_tanh, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(0.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_tanh(a);
    cr_assert_float_eq(tensor_item(r), 0.0, 1e-12);
    tensor_backward(r);
    /* d tanh(x)/dx at x=0 = 1 - 0^2 = 1 */
    cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12,
        "d tanh(0)/dx should be 1.0 (got %.6f)", param_grad_item_at(0, 0));
}

Test(tape_core_elementwise_softplus, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(0.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_softplus(a);
    /* softplus(0) = log(2) */
    cr_assert_float_eq(tensor_item(r), log(2.0), 1e-12);
    tensor_backward(r);
    /* d softplus(0)/dx = sigmoid(0) = 0.5 */
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, 1e-12,
        "d softplus(0)/dx should be 0.5 (got %.6f)", param_grad_item_at(0, 0));
}

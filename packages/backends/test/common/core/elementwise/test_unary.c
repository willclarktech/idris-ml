/* Criterion suite for tape unary elementwise ops.
 * Covers neg, abs, exp, log, sqrt. */

#include <criterion/criterion.h>
#include <math.h>
#include "../../../../backend.h"
#include "../../test_helpers.h"

Test(tape_core_elementwise_neg, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(7.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_neg(a);
    cr_assert_float_eq(tensor_item(r), -7.0, 1e-12);
    tensor_backward(r);
    cr_assert_float_eq(param_grad_item_at(0, 0), -1.0, 1e-12,
        "d(-x)/dx should be -1 (got %.6f)", param_grad_item_at(0, 0));
}

Test(tape_core_elementwise_abs, forward_backward_pos_neg) {
    param_clear();
    TensorHandle a = tensor_create_scalar(-3.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_abs(a);
    cr_assert_float_eq(tensor_item(r), 3.0, 1e-12);
    tensor_backward(r);
    cr_assert_float_eq(param_grad_item_at(0, 0), -1.0, 1e-12,
        "d|x|/dx at x=-3 should be -1 (got %.6f)", param_grad_item_at(0, 0));
}

Test(tape_core_elementwise_exp, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(1.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_exp(a);
    cr_assert_float_eq(tensor_item(r), exp(1.0), TEST_TOL_TIGHT);
    tensor_backward(r);
    /* d(exp(x))/dx at x=1 = exp(1) */
    cr_assert_float_eq(param_grad_item_at(0, 0), exp(1.0), TEST_TOL_TIGHT,
        "d(exp(x))/dx at x=1 should be e (got %.6f)", param_grad_item_at(0, 0));
}

Test(tape_core_elementwise_log, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(2.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_log(a);
    cr_assert_float_eq(tensor_item(r), log(2.0), TEST_TOL_TIGHT);
    tensor_backward(r);
    /* d(log(x))/dx at x=2 = 1/2 */
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, TEST_TOL_TIGHT,
        "d(log(x))/dx at x=2 should be 0.5 (got %.6f)", param_grad_item_at(0, 0));
}

Test(tape_core_elementwise_sqrt, forward_backward) {
    param_clear();
    TensorHandle a = tensor_create_scalar(4.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_sqrt(a);
    cr_assert_float_eq(tensor_item(r), 2.0, TEST_TOL_TIGHT);
    tensor_backward(r);
    /* d(sqrt(x))/dx at x=4 = 1/(2*sqrt(4)) = 0.25 */
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.25, TEST_TOL_TIGHT,
        "d(sqrt(x))/dx at x=4 should be 0.25 (got %.6f)", param_grad_item_at(0, 0));
}

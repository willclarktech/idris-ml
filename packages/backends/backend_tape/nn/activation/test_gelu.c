/* Criterion suite for tensor_gelu (forward + backward).
 *
 * GELU (tanh approximation):
 *   gelu(x) = x * 0.5 * (1 + tanh(c * (x + 0.044715 * x^3))),  c = sqrt(2/pi)
 *
 * Closed-form anchors used:
 *   gelu(0)  = 0
 *   gelu'(0) = 0.5 * (1 + tanh(0)) + 0.5 * 0 * dtanh = 0.5
 *
 * Closes the W3/W4 OP_GELU coverage gap on tape + mlx (probed by
 * scripts/coverage-gap-probe.sh prior to this commit).
 */

#include <criterion/criterion.h>
#include <math.h>
#include "backend.h"
#include "test_helpers.h"

static double gelu_ref(double x) {
    double c = 0.7978845608028654;
    double inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

static double gelu_grad_ref(double x) {
    double c = 0.7978845608028654;
    double inner = c * (x + 0.044715 * x * x * x);
    double t = tanh(inner);
    double dtdx = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * dtdx;
}

Test(nn_activation_gelu, forward_backward_at_zero) {
    param_clear();
    TensorHandle a = tensor_create_scalar(0.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_gelu(a);
    cr_assert_float_eq(tensor_item(r), 0.0, TEST_TOL_TIGHT,
        "gelu(0) should be 0 (got %.9f)", tensor_item(r));
    tensor_backward(r);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, TEST_TOL_TIGHT,
        "d gelu(0)/dx should be 0.5 (got %.9f)", param_grad_item_at(0, 0));
}

Test(nn_activation_gelu, forward_backward_at_one) {
    param_clear();
    TensorHandle a = tensor_create_scalar(1.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_gelu(a);
    cr_assert_float_eq(tensor_item(r), gelu_ref(1.0), TEST_TOL_RELAXED,
        "gelu(1) should match reference (got %.9f vs %.9f)",
        tensor_item(r), gelu_ref(1.0));
    tensor_backward(r);
    cr_assert_float_eq(param_grad_item_at(0, 0), gelu_grad_ref(1.0), TEST_TOL_RELAXED,
        "d gelu(1)/dx should match reference (got %.9f vs %.9f)",
        param_grad_item_at(0, 0), gelu_grad_ref(1.0));
}

Test(nn_activation_gelu, forward_negative_input) {
    /* GELU is asymmetric: gelu(-x) ≠ -gelu(x). At x=-1 the output is small
     * and negative (the tanh damp factor). Verify the sign is correct. */
    param_clear();
    TensorHandle a = tensor_create_scalar(-1.0, 1);
    param_register("a", a);
    TensorHandle r = tensor_gelu(a);
    double expected = gelu_ref(-1.0);
    cr_assert(expected < 0.0, "gelu(-1) reference should be negative");
    cr_assert_float_eq(tensor_item(r), expected, TEST_TOL_RELAXED,
        "gelu(-1) should match reference (got %.9f vs %.9f)",
        tensor_item(r), expected);
}

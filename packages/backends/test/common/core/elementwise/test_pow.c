/* Criterion suite for tape `tensor_pow`. */

#include <criterion/criterion.h>
#include <math.h>
#include "../../../../backend.h"
#include "test_helpers.h"

Test(core_elementwise_pow, forward_scalar) {
    TensorHandle a = tensor_create_scalar(2.0, 0);
    TensorHandle b = tensor_create_scalar(3.0, 0);
    TensorHandle c = tensor_pow(a, b);
    cr_assert_float_eq(tensor_item(c), 8.0, TEST_TOL_TIGHT);
}

Test(core_elementwise_pow, backward_scalar) {
    /* c = a^b; dc/da = b*a^(b-1) = 3*2^2 = 12
                dc/db = log(a)*a^b = ln(2)*8 */
    param_clear();
    TensorHandle a = tensor_create_scalar(2.0, 1);
    TensorHandle b = tensor_create_scalar(3.0, 1);
    param_register("a", a);
    param_register("b", b);
    TensorHandle c = tensor_pow(a, b);
    tensor_backward(c);
    cr_assert_float_eq(param_grad_item_at(0, 0), 12.0, TEST_TOL_TIGHT,
        "d(a^b)/da at a=2,b=3 should be b*a^(b-1)=12.0 (got %.6f)", param_grad_item_at(0, 0));
    cr_assert_float_eq(param_grad_item_at(1, 0), log(2.0) * 8.0, TEST_TOL_TIGHT,
        "d(a^b)/db at a=2,b=3 should be ln(2)*8 (got %.6f)", param_grad_item_at(1, 0));
}

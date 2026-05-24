/* Criterion suite for tape `tensor_stack_from_array`.
 *
 * Stacks N scalar tensors into a length-N vector. Backward distributes
 * the upstream vector grad elementwise back to each constituent scalar's
 * grad[0].
 *
 * RED: dispatch NULL for OP_STACK → backward leaves scalar grads at 0
 * → first assertion (d_a[0]) fires.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"

Test(linear_concat_stack_from_array, backward_distributes) {
    param_clear();
    TensorHandle a = tensor_create_scalar(1.0, 1);
    TensorHandle b = tensor_create_scalar(2.0, 1);
    TensorHandle c = tensor_create_scalar(3.0, 1);
    param_register("a", a);
    param_register("b", b);
    param_register("c", c);

    TensorHandle* arr = (TensorHandle*)malloc(3 * sizeof(TensorHandle));
    arr[0] = a; arr[1] = b; arr[2] = c;
    /* tensor_stack_from_array consumes arr (frees it internally). */
    TensorHandle v = tensor_stack_from_array(arr, 3, /*dim=*/0);

    /* Forward: v = [1, 2, 3] */
    cr_assert_float_eq(tensor_item_1d(v, 0), 1.0, 1e-12);
    cr_assert_float_eq(tensor_item_1d(v, 1), 2.0, 1e-12);
    cr_assert_float_eq(tensor_item_1d(v, 2), 3.0, 1e-12);

    /* loss = sum(v) → d_v = [1,1,1] → d_a = d_b = d_c = 1 */
    TensorHandle loss = tensor_sum(v);
    tensor_backward(loss);
    cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12, "d_a");
    cr_assert_float_eq(param_grad_item_at(1, 0), 1.0, 1e-12, "d_b");
    cr_assert_float_eq(param_grad_item_at(2, 0), 1.0, 1e-12, "d_c");
}

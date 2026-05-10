/* Criterion suite for tape `tensor_max_pool2d` (Phase 1d.2.b).
 *
 * input [1, 2, 2] = [[1,2],[3,4]], kH=kW=2, strideH=strideW=1 → oH=oW=1.
 * Forward: out=[4] (max). Backward sum-loss: d_in[3]=1, others 0.
 */

#include <criterion/criterion.h>
#include "../../../backend.h"

Test(tape_conv_max_pool2d, forward_and_backward) {
    param_clear();
    double in_data[4] = {1.0, 2.0, 3.0, 4.0};
    int sh[3] = {1, 2, 2};
    TensorHandle in = tensor_create(in_data, sh, 3, 1);
    param_register("in", in);

    TensorHandle out = tensor_max_pool2d(in, 2, 2, 1, 1);
    cr_assert_float_eq(tensor_item_1d(out, 0), 4.0, 1e-12);

    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12, "winner");
}

/* Criterion suite for tape `tensor_max_pool1d`.
 *
 * input [1, 4] = [3, 1, 4, 2], kL=2, stride=2 → out = [3, 4] (wins at idx 0, 2).
 * Backward: d_in[winner_idx] += d_out, others = 0.
 *
 * RED: dispatch NULL → grads zero → d_in[0] expected 1.0 fires.
 */

#include <criterion/criterion.h>
#include "backend.h"

Test(conv_max_pool1d, forward_and_backward) {
    param_clear();
    double in_data[4] = {3.0, 1.0, 4.0, 2.0};
    int sh[2] = {1, 4};
    TensorHandle in = tensor_create(in_data, sh, 2, 1);
    param_register("in", in);

    TensorHandle out = tensor_max_pool1d(in, /*kL=*/2, /*stride=*/2);
    cr_assert_float_eq(tensor_item_1d(out, 0), 3.0, 1e-12);
    cr_assert_float_eq(tensor_item_1d(out, 1), 4.0, 1e-12);

    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    /* Winners: idx 0 and idx 2 each get 1.0; idx 1 and 3 get 0. */
    cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12, "d_in[0]");
    cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, 1e-12, "d_in[1]");
    cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "d_in[2]");
    cr_assert_float_eq(param_grad_item_at(0, 3), 0.0, 1e-12, "d_in[3]");
}

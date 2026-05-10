/* Criterion suite for tape `tensor_conv2d_batched`.
 *
 * input [B=1, inC=1, H=2, W=2] = [[1,2],[3,4]] (single sample),
 * kernel [outC=1, inC=1, kH=2, kW=2] = [[1,1],[1,1]], no bias, pad=0, stride=1.
 * Forward: B=1 reduces to single-sample conv2d; out[0,0,0,0] = 1+2+3+4 = 10.
 * Backward sum-loss: d_in[i] = 1, d_k[i] = in[i].
 *
 * RED: dispatch NULL → d_in[0] expected 1 fires.
 */

#include <criterion/criterion.h>
#include "../../../backend.h"

Test(tape_conv_conv2d_batched, forward_and_backward) {
    param_clear();
    double in_data[4] = {1.0, 2.0, 3.0, 4.0};
    double k_data[4]  = {1.0, 1.0, 1.0, 1.0};
    int sh_in[4] = {1, 1, 2, 2};
    int sh_k[4]  = {1, 1, 2, 2};
    TensorHandle in = tensor_create(in_data, sh_in, 4, 1);
    TensorHandle k  = tensor_create(k_data,  sh_k,  4, 1);
    param_register("in", in);
    param_register("k",  k);

    TensorHandle out = tensor_conv2d_batched(in, k, (TensorHandle)0, 0, 0, 1, 1);
    cr_assert_float_eq(tensor_item_1d(out, 0), 10.0, 1e-12);

    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12, "d_in[%d]", i);
    double exp_k[4] = {1.0, 2.0, 3.0, 4.0};
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
}

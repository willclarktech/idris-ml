/* Criterion suite for tape `tensor_conv1d`.
 *
 * input [1, 4] = [1, 2, 3, 4], kernel [1, 1, 2] = [0.5, 0.5], bias=NULL,
 * pad=0, stride=1 → oL = (4 + 0 - 2)/1 + 1 = 3.
 *   out[0,0] = 1*0.5 + 2*0.5 = 1.5
 *   out[0,1] = 2*0.5 + 3*0.5 = 2.5
 *   out[0,2] = 3*0.5 + 4*0.5 = 3.5
 *
 * Backward sum-loss: d_out=[1,1,1].
 * d_input[i] = sum over (oc, kl) where il=i: d_out[oc, ol] * kernel[oc, ic, kl]
 *   d_in[0] = 0.5 (only ol=0, kl=0 hits il=0)
 *   d_in[1] = 0.5 + 0.5 = 1.0   (ol=0 kl=1, ol=1 kl=0)
 *   d_in[2] = 0.5 + 0.5 = 1.0   (ol=1 kl=1, ol=2 kl=0)
 *   d_in[3] = 0.5
 * d_kernel[oc=0, ic=0, kl=0] = sum_ol d_out[ol] * in[ol]    = 1+2+3 = 6
 * d_kernel[oc=0, ic=0, kl=1] = sum_ol d_out[ol] * in[ol+1]  = 2+3+4 = 9
 *
 * RED: dispatch NULL → grads zero → d_in[0] expected 0.5 fires.
 */

#include <criterion/criterion.h>
#include "../../../backend.h"

Test(conv_conv1d, forward_and_backward) {
    param_clear();
    double in_data[4] = {1.0, 2.0, 3.0, 4.0};
    double k_data[2]  = {0.5, 0.5};
    int sh_in[2] = {1, 4};
    int sh_k[3]  = {1, 1, 2};
    TensorHandle in = tensor_create(in_data, sh_in, 2, 1);
    TensorHandle k  = tensor_create(k_data,  sh_k,  3, 1);
    param_register("in", in);
    param_register("k",  k);

    TensorHandle out = tensor_conv1d(in, k, /*bias=*/(TensorHandle)0, /*pad=*/0, /*stride=*/1);
    cr_assert_float_eq(tensor_item_1d(out, 0), 1.5, 1e-12);
    cr_assert_float_eq(tensor_item_1d(out, 1), 2.5, 1e-12);
    cr_assert_float_eq(tensor_item_1d(out, 2), 3.5, 1e-12);

    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, 1e-12, "d_in[0]");
    cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12, "d_in[1]");
    cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "d_in[2]");
    cr_assert_float_eq(param_grad_item_at(0, 3), 0.5, 1e-12, "d_in[3]");
    cr_assert_float_eq(param_grad_item_at(1, 0), 6.0, 1e-12, "d_k[0]");
    cr_assert_float_eq(param_grad_item_at(1, 1), 9.0, 1e-12, "d_k[1]");
}

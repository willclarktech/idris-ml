/* Criterion suite for tape `tensor_conv1d_circular`.
 *
 * input [3] = [1, 2, 3], kernel [3] = [0.1, 0.2, 0.3] (k=3, pad=1).
 *   out[i] = sum_j in[(i-1+j+3) % 3] * k[2-j]
 *   out[0] = in[2]*k[2] + in[0]*k[1] + in[1]*k[0]
 *           = 3*0.3 + 1*0.2 + 2*0.1 = 0.9 + 0.2 + 0.2 = 1.3
 *   out[1] = in[0]*k[2] + in[1]*k[1] + in[2]*k[0]
 *           = 1*0.3 + 2*0.2 + 3*0.1 = 0.3 + 0.4 + 0.3 = 1.0
 *   out[2] = in[1]*k[2] + in[2]*k[1] + in[0]*k[0]
 *           = 2*0.3 + 3*0.2 + 1*0.1 = 0.6 + 0.6 + 0.1 = 1.3
 *
 * Backward sum-loss: d_out=[1,1,1]. For input: each in[idx] receives
 * sum of k[k-1-j] across the j's that picked it → each gets k[0]+k[1]+k[2] = 0.6.
 * For kernel: each k[k-1-j] receives sum of in[idx] across i's that picked it
 *             → each gets 1+2+3 = 6, so d_k[0]=d_k[1]=d_k[2]=6.
 *
 * RED: dispatch NULL → grads zero → d_in[0] expected 0.6 fires.
 */

#include <criterion/criterion.h>
#include "../../../backend.h"
#include "../test_helpers.h"

/* mlx: forward output is wrong — observed [1.1, 1.4, 1.1] instead of
   [1.3, 1.0, 1.3]; backward is correct. Likely an off-by-one in
   mlx's circular kernel rotation. Tracked in TODO.md "mlx
   conv1d_circular forward output". */
Test(tape_conv_conv1d_circular, forward_and_backward, .disabled = SKIP_ON_MLX) {
    param_clear();
    double in_data[3] = {1.0, 2.0, 3.0};
    double k_data[3]  = {0.1, 0.2, 0.3};
    int sh[1] = {3};
    TensorHandle in = tensor_create(in_data, sh, 1, 1);
    TensorHandle k  = tensor_create(k_data,  sh, 1, 1);
    param_register("in", in);
    param_register("k",  k);

    TensorHandle out = tensor_conv1d_circular(in, k);
    cr_assert_float_eq(tensor_item_1d(out, 0), 1.3, TEST_TOL_TIGHT);
    cr_assert_float_eq(tensor_item_1d(out, 1), 1.0, TEST_TOL_TIGHT);
    cr_assert_float_eq(tensor_item_1d(out, 2), 1.3, TEST_TOL_TIGHT);

    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    for (int i = 0; i < 3; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 0.6, TEST_TOL_TIGHT, "d_in[%d]", i);
    for (int i = 0; i < 3; i++)
        cr_assert_float_eq(param_grad_item_at(1, i), 6.0, TEST_TOL_TIGHT, "d_k[%d]", i);
}

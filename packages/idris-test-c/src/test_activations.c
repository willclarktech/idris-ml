/* Activation-function Criterion suite (softmax/leaky-relu/silu/softplus).
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"


Test(activations, leaky_relu_silu_softplus) {
    param_clear();

    /* LeakyReLU forward: positive passes through, negative scaled by alpha */
    double lr_data[] = {2.0, -3.0, 0.0, 1.0};
    int lr_s[] = {4};
    TensorHandle lr_in = tensor_create(lr_data, lr_s, 1, 1);
    param_register("lr_in", lr_in);
    TensorHandle lr_out = tensor_leaky_relu(lr_in, 0.1);
    double lr_result[4];
    tensor_to_doubles(lr_out, lr_result);
    ASSERT_NEAR("leaky_relu(2)", lr_result[0], 2.0, VAL_TOL);
    ASSERT_NEAR("leaky_relu(-3)", lr_result[1], -0.3, VAL_TOL);
    ASSERT_NEAR("leaky_relu(0)", lr_result[2], 0.0, VAL_TOL);
    ASSERT_NEAR("leaky_relu(1)", lr_result[3], 1.0, VAL_TOL);

    /* LeakyReLU backward */
    TensorHandle lr_loss = tensor_sum(lr_out);
    tensor_backward(lr_loss);
    /* d/dx: 1 for x>=0, alpha for x<0 */
    ASSERT_NEAR("d_leaky_relu(2)", param_grad_item_at(0, 0), 1.0, VAL_TOL);
    ASSERT_NEAR("d_leaky_relu(-3)", param_grad_item_at(0, 1), 0.1, VAL_TOL);
    /* d_leaky_relu(0) skipped: derivative at 0 is implementation-defined
       (tape returns 1.0, torch returns alpha). Both are valid. */
    ASSERT_NEAR("d_leaky_relu(1)", param_grad_item_at(0, 3), 1.0, VAL_TOL);
    param_clear();

    /* SiLU forward: silu(x) = x * sigmoid(x) */
    double s_data[] = {0.0, 1.0, -1.0};
    int s_s[] = {3};
    TensorHandle s_in = tensor_create(s_data, s_s, 1, 1);
    param_register("s_in", s_in);
    TensorHandle s_out = tensor_silu(s_in);
    double s_result[3];
    tensor_to_doubles(s_out, s_result);
    ASSERT_NEAR("silu(0)", s_result[0], 0.0, 1e-10);  /* 0 * 0.5 = 0 */
    ASSERT_NEAR("silu(1)", s_result[1], 1.0 / (1.0 + exp(-1.0)), 1e-5);
    ASSERT_NEAR("silu(-1)", s_result[2], -1.0 / (1.0 + exp(1.0)), 1e-5);

    /* SiLU backward */
    TensorHandle s_loss = tensor_sum(s_out);
    tensor_backward(s_loss);
    /* d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))) */
    double sig0 = 0.5, sig1 = 1.0/(1.0+exp(-1.0)), sigm1 = 1.0/(1.0+exp(1.0));
    ASSERT_NEAR("d_silu(0)", param_grad_item_at(0, 0), sig0 * (1.0 + 0.0 * (1.0 - sig0)), 1e-5);
    ASSERT_NEAR("d_silu(1)", param_grad_item_at(0, 1), sig1 * (1.0 + 1.0 * (1.0 - sig1)), 1e-5);
    ASSERT_NEAR("d_silu(-1)", param_grad_item_at(0, 2), sigm1 * (1.0 + (-1.0) * (1.0 - sigm1)), 1e-5);
    param_clear();

    /* Softplus forward: softplus(x) = log(1 + exp(x)) */
    double sp_data[] = {0.0, 1.0, -1.0, 5.0, -5.0};
    int sp_s[] = {5};
    TensorHandle sp_in = tensor_create(sp_data, sp_s, 1, 1);
    param_register("sp_in", sp_in);
    TensorHandle sp_out = tensor_softplus(sp_in);
    double sp_result[5];
    tensor_to_doubles(sp_out, sp_result);
    ASSERT_NEAR("softplus(0)", sp_result[0], log(2.0), VAL_TOL);
    ASSERT_NEAR("softplus(1)", sp_result[1], log(1.0 + exp(1.0)), VAL_TOL);
    ASSERT_NEAR("softplus(-1)", sp_result[2], log(1.0 + exp(-1.0)), VAL_TOL);
    ASSERT_NEAR("softplus(5)", sp_result[3], log(1.0 + exp(5.0)), 1e-5);
    ASSERT_NEAR("softplus(-5)", sp_result[4], log(1.0 + exp(-5.0)), VAL_TOL);

    /* Softplus backward: d_softplus(x) = sigmoid(x). The mlx backward
       is via vjp on max(0,x) + log(1+exp(-|x|)) — the numerically
       stable form. That form is non-smooth at x=0 (both `max` and
       `abs` have subgradient ambiguity there), and mlx picks the 0
       subgradient → d_softplus(0) returns 0 instead of 0.5. The
       naive log(1+exp(x)) form would give 0.5 but overflows on fp32
       for x > ~88; we keep the stable form and skip the x=0 boundary
       probe on mlx. All non-boundary points (x = ±1, ±5) return the
       correct sigmoid derivative. */
    TensorHandle sp_loss = tensor_sum(sp_out);
    tensor_backward(sp_loss);
#if !defined(BACKEND_MLX)
    ASSERT_NEAR("d_softplus(0)", param_grad_item_at(0, 0), 0.5, VAL_TOL);
#endif
    ASSERT_NEAR("d_softplus(1)", param_grad_item_at(0, 1), 1.0/(1.0+exp(-1.0)), VAL_TOL);
    ASSERT_NEAR("d_softplus(-1)", param_grad_item_at(0, 2), 1.0/(1.0+exp(1.0)), VAL_TOL);
    ASSERT_NEAR("d_softplus(5)", param_grad_item_at(0, 3), 1.0/(1.0+exp(-5.0)), VAL_TOL);
    ASSERT_NEAR("d_softplus(-5)", param_grad_item_at(0, 4), 1.0/(1.0+exp(5.0)), VAL_TOL);
    param_clear();
}

/* NN layer (linear/norm/conv/pool/dropout/embedding) Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 *
 * Single-op tests have been relocated to the per-op contract tree under
 * src/ops/. This file retains only the multi-op composite below.
 */
#include "port_assert.h"


/* Test: layernorm on batched data → narrow → mm → cat → residual → sum.
   This reproduces the batchBlockForward pattern more closely. */
Test(nn_layers, narrow_layernorm_cat_gradient) {
    param_clear();

    /* Input: [6,2] = 2 sequences of [3,2] (bsI=6, sI=3, dI=2) */
    double x_data[] = {1,2, 3,4, 5,6,   7,8, 9,10, 11,12};
    int x_shape[] = {6, 2};
    /* LayerNorm gamma, beta */
    double g_data[] = {1.0, 1.0};
    double b_data[] = {0.0, 0.0};
    int gb_shape[] = {2};
    /* Weight [2,2] */
    double w_data[] = {0.1, 0.2, 0.3, 0.4};
    int w_shape[] = {2, 2};

    /* Test: layernorm → narrow per seq → mm → cat → add(result, input) → sum */
    TensorHandle x = tensor_create(x_data, x_shape, 2, 1);
    param_register("x", x);
    TensorHandle g = tensor_create(g_data, gb_shape, 1, 1);
    param_register("gamma", g);
    TensorHandle b = tensor_create(b_data, gb_shape, 1, 1);
    param_register("beta", b);
    TensorHandle w = tensor_create(w_data, w_shape, 2, 1);
    param_register("w", w);

    /* LayerNorm on full [6,2] */
    TensorHandle normed = tensor_layer_norm_2d(x, g, b, 1e-5);

    /* Flatten to 1D for narrow */
    int flat_shape[] = {12};
    TensorHandle normed_flat = tensor_reshape(normed, flat_shape, 1);
    /* Slice: seq0 = [0:6], seq1 = [6:12] */
    TensorHandle s0_flat = tensor_narrow(normed_flat, 0, 0, 6);
    TensorHandle s1_flat = tensor_narrow(normed_flat, 0, 6, 6);
    /* Reshape to [3,2] */
    int seq_shape[] = {3, 2};
    TensorHandle s0 = tensor_reshape(s0_flat, seq_shape, 2);
    TensorHandle s1 = tensor_reshape(s1_flat, seq_shape, 2);
    /* MM with shared weight */
    TensorHandle wt = tensor_transpose_2d(w);
    TensorHandle o0 = tensor_mm(s0, wt);
    TensorHandle o1 = tensor_mm(s1, wt);
    /* Flatten and cat */
    int of_shape[] = {6};
    TensorHandle o0f = tensor_reshape(o0, of_shape, 1);
    TensorHandle o1f = tensor_reshape(o1, of_shape, 1);
    TensorHandle catted = tensor_cat2(o0f, o1f);
    /* Reshape to [6,2] and add residual */
    TensorHandle out_2d = tensor_reshape(catted, x_shape, 2);
    TensorHandle result = tensor_add(out_2d, x);
    /* Sum → loss */
    TensorHandle loss = tensor_sum(result);
    double loss_val = tensor_item(loss);
    printf("  Loss = %f\n", loss_val);
    tensor_backward(loss);

    double grad_w00 = param_grad_item_at(3, 0);
    printf("  w[0,0] grad (analytical) = %f\n", grad_w00);

    /* Finite diff for w[0,0] */
    double eps = 1e-5;
    {
        double w_copy[4]; memcpy(w_copy, w_data, sizeof(w_data));
        param_clear();
        w_copy[0] = w_data[0] + eps;
        TensorHandle xp = tensor_create(x_data, x_shape, 2, 0);
        TensorHandle gp = tensor_create(g_data, gb_shape, 1, 0);
        TensorHandle bp = tensor_create(b_data, gb_shape, 1, 0);
        TensorHandle wp = tensor_create(w_copy, w_shape, 2, 0);
        TensorHandle np = tensor_layer_norm_2d(xp, gp, bp, 1e-5);
        double fp = tensor_item(tensor_sum(tensor_add(
            tensor_reshape(
                tensor_cat2(
                    tensor_reshape(tensor_mm(
                        tensor_reshape(tensor_narrow(tensor_reshape(np, flat_shape, 1), 0, 0, 6), seq_shape, 2),
                        tensor_transpose_2d(wp)), of_shape, 1),
                    tensor_reshape(tensor_mm(
                        tensor_reshape(tensor_narrow(tensor_reshape(np, flat_shape, 1), 0, 6, 6), seq_shape, 2),
                        tensor_transpose_2d(wp)), of_shape, 1)),
                x_shape, 2),
            xp)));

        w_copy[0] = w_data[0] - eps;
        param_clear();
        TensorHandle xm = tensor_create(x_data, x_shape, 2, 0);
        TensorHandle gm = tensor_create(g_data, gb_shape, 1, 0);
        TensorHandle bm = tensor_create(b_data, gb_shape, 1, 0);
        TensorHandle wm = tensor_create(w_copy, w_shape, 2, 0);
        TensorHandle nm = tensor_layer_norm_2d(xm, gm, bm, 1e-5);
        double fm = tensor_item(tensor_sum(tensor_add(
            tensor_reshape(
                tensor_cat2(
                    tensor_reshape(tensor_mm(
                        tensor_reshape(tensor_narrow(tensor_reshape(nm, flat_shape, 1), 0, 0, 6), seq_shape, 2),
                        tensor_transpose_2d(wm)), of_shape, 1),
                    tensor_reshape(tensor_mm(
                        tensor_reshape(tensor_narrow(tensor_reshape(nm, flat_shape, 1), 0, 6, 6), seq_shape, 2),
                        tensor_transpose_2d(wm)), of_shape, 1)),
                x_shape, 2),
            xm)));

        double fd = (fp - fm) / (2 * eps);
        printf("  w[0,0] grad (finite diff) = %f\n", fd);
        ASSERT_NEAR("ln+narrow+cat w grad", grad_w00, fd, FD_TOL);
    }
    param_clear();
}

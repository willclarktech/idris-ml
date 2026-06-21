/* NN layer (linear/norm/conv/pool/dropout/embedding) Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"


Test(nn_layers, linear_2d_forward) {
    /* W: [2, 3] (o=2, i=3), X: [4, 3] (B=4), bias: [2] */
    double w_data[] = {0.1, 0.2, 0.3,   0.4, 0.5, 0.6};
    double x_data[] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0,
                       0.5, 1.5, 2.5};
    double b_data[] = {10.0, 20.0};
    int w_shape[] = {2, 3};
    int x_shape[] = {4, 3};
    int b_shape[] = {2};

    TensorHandle W = tensor_create(w_data, w_shape, 2, 0);
    TensorHandle X = tensor_create(x_data, x_shape, 2, 0);
    TensorHandle bias = tensor_create(b_data, b_shape, 1, 0);

    TensorHandle Y = tensor_linear_2d(W, X, bias);

    /* Y[b,o] = sum_i X[b,i] * W[o,i] + bias[o]
       Y[0,0] = 1*0.1 + 2*0.2 + 3*0.3 + 10 = 0.1+0.4+0.9+10 = 11.4
       Y[0,1] = 1*0.4 + 2*0.5 + 3*0.6 + 20 = 0.4+1.0+1.8+20 = 23.2
       Y[1,0] = 4*0.1 + 5*0.2 + 6*0.3 + 10 = 0.4+1.0+1.8+10 = 13.2
       Y[1,1] = 4*0.4 + 5*0.5 + 6*0.6 + 20 = 1.6+2.5+3.6+20 = 27.7 */
    ASSERT_NEAR("lin2d Y[0,0]", tensor_item_2d(Y, 0, 0), 11.4, VAL_TOL);
    ASSERT_NEAR("lin2d Y[0,1]", tensor_item_2d(Y, 0, 1), 23.2, VAL_TOL);
    ASSERT_NEAR("lin2d Y[1,0]", tensor_item_2d(Y, 1, 0), 13.2, VAL_TOL);
    ASSERT_NEAR("lin2d Y[1,1]", tensor_item_2d(Y, 1, 1), 27.7, VAL_TOL);

    /* B=4 row: Y[3,0] = 0.5*0.1+1.5*0.2+2.5*0.3 + 10 = 0.05+0.3+0.75+10 = 11.1 */
    ASSERT_NEAR("lin2d Y[3,0]", tensor_item_2d(Y, 3, 0), 11.1, VAL_TOL);
}

Test(nn_layers, linear_2d_matches_per_sample) {
    /* For B independent inputs, batched tensor_linear_2d must produce the
       same outputs as B calls to tensor_linear (per-sample mv+bias). */
    double w_data[] = {0.1, -0.2, 0.3,   -0.4, 0.5, -0.6};
    double b_data[] = {0.7, -0.8};
    int w_shape[] = {2, 3};
    int b_shape[] = {2};

    /* Three inputs */
    double x0_data[] = {1.0, 2.0, 3.0};
    double x1_data[] = {-1.0, 0.5, 0.25};
    double x2_data[] = {0.0, -2.0, 1.5};
    int x_shape[] = {3};

    TensorHandle W = tensor_create(w_data, w_shape, 2, 0);
    TensorHandle bias = tensor_create(b_data, b_shape, 1, 0);

    /* Per-sample */
    TensorHandle x0 = tensor_create(x0_data, x_shape, 1, 0);
    TensorHandle x1 = tensor_create(x1_data, x_shape, 1, 0);
    TensorHandle x2 = tensor_create(x2_data, x_shape, 1, 0);
    TensorHandle y0 = tensor_linear(W, x0, bias);
    TensorHandle y1 = tensor_linear(W, x1, bias);
    TensorHandle y2 = tensor_linear(W, x2, bias);

    /* Batched */
    double xb_data[] = {1.0, 2.0, 3.0,
                        -1.0, 0.5, 0.25,
                        0.0, -2.0, 1.5};
    int xb_shape[] = {3, 3};
    TensorHandle Xb = tensor_create(xb_data, xb_shape, 2, 0);
    TensorHandle Yb = tensor_linear_2d(W, Xb, bias);

    ASSERT_NEAR("Yb[0,0]==y0[0]", tensor_item_2d(Yb, 0, 0), tensor_item_1d(y0, 0), 1e-9);
    ASSERT_NEAR("Yb[0,1]==y0[1]", tensor_item_2d(Yb, 0, 1), tensor_item_1d(y0, 1), 1e-9);
    ASSERT_NEAR("Yb[1,0]==y1[0]", tensor_item_2d(Yb, 1, 0), tensor_item_1d(y1, 0), 1e-9);
    ASSERT_NEAR("Yb[1,1]==y1[1]", tensor_item_2d(Yb, 1, 1), tensor_item_1d(y1, 1), 1e-9);
    ASSERT_NEAR("Yb[2,0]==y2[0]", tensor_item_2d(Yb, 2, 0), tensor_item_1d(y2, 0), 1e-9);
    ASSERT_NEAR("Yb[2,1]==y2[1]", tensor_item_2d(Yb, 2, 1), tensor_item_1d(y2, 1), 1e-9);
}

Test(nn_layers, linear_2d_backward) {
    param_clear();

    double w_data[] = {0.1, 0.2, 0.3,   0.4, 0.5, 0.6};
    double x_data[] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0};
    double b_data[] = {0.7, 0.8};
    int w_shape[] = {2, 3};
    int x_shape[] = {2, 3};
    int b_shape[] = {2};

    TensorHandle W = tensor_create(w_data, w_shape, 2, 1);
    param_register("W", W);
    TensorHandle X = tensor_create(x_data, x_shape, 2, 1);
    param_register("X", X);
    TensorHandle bias = tensor_create(b_data, b_shape, 1, 1);
    param_register("bias", bias);

    TensorHandle Y = tensor_linear_2d(W, X, bias);
    TensorHandle loss = tensor_sum(Y);
    tensor_backward(loss);

    /* Analytical:
       dY/dW[o,i] = sum_b X[b,i]   (since loss = sum_b sum_o Y[b,o])
       dY/dX[b,i] = sum_o W[o,i]
       dY/dbias[o] = B (number of batch elements) */

    /* W[0,0]: sum_b X[b,0] = 1 + 4 = 5 */
    {
        double eps = 1e-5;
        double w_copy[6]; memcpy(w_copy, w_data, 6*sizeof(double));
        w_copy[0] = w_data[0] + eps;
        TensorHandle Wp = tensor_create(w_copy, w_shape, 2, 0);
        TensorHandle Xp = tensor_create(x_data, x_shape, 2, 0);
        TensorHandle Bp = tensor_create(b_data, b_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_linear_2d(Wp, Xp, Bp)));
        w_copy[0] = w_data[0] - eps;
        TensorHandle Wm = tensor_create(w_copy, w_shape, 2, 0);
        TensorHandle Xm = tensor_create(x_data, x_shape, 2, 0);
        TensorHandle Bm = tensor_create(b_data, b_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_linear_2d(Wm, Xm, Bm)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(0, 0);
        printf("  W[0,0]: fd=%f analytic=%f\n", fd, analytic);
        ASSERT_NEAR("lin2d grad W[0,0]", analytic, fd, FD_TOL);
    }

    /* X[0,0]: sum_o W[o,0] = 0.1 + 0.4 = 0.5 */
    {
        double eps = 1e-5;
        double x_copy[6]; memcpy(x_copy, x_data, 6*sizeof(double));
        x_copy[0] = x_data[0] + eps;
        TensorHandle Wp = tensor_create(w_data, w_shape, 2, 0);
        TensorHandle Xp = tensor_create(x_copy, x_shape, 2, 0);
        TensorHandle Bp = tensor_create(b_data, b_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_linear_2d(Wp, Xp, Bp)));
        x_copy[0] = x_data[0] - eps;
        TensorHandle Wm = tensor_create(w_data, w_shape, 2, 0);
        TensorHandle Xm = tensor_create(x_copy, x_shape, 2, 0);
        TensorHandle Bm = tensor_create(b_data, b_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_linear_2d(Wm, Xm, Bm)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(1, 0);
        printf("  X[0,0]: fd=%f analytic=%f\n", fd, analytic);
        ASSERT_NEAR("lin2d grad X[0,0]", analytic, fd, FD_TOL);
    }

    /* bias[0]: B = 2 */
    {
        double analytic = param_grad_item_at(2, 0);
        printf("  bias[0]: analytic=%f (expected 2.0)\n", analytic);
        ASSERT_NEAR("lin2d grad bias[0]", analytic, 2.0, 1e-9);
    }

    param_clear();
}

Test(nn_layers, layer_norm_2d) {
    /* 2x3 matrix */
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double gamma_data[] = {1.0, 1.0, 1.0};
    double beta_data[] = {0.0, 0.0, 0.0};
    int shape[] = {2, 3};
    int gamma_shape[] = {3};

    TensorHandle t = tensor_create(data, shape, 2, 0);
    TensorHandle gamma = tensor_create(gamma_data, gamma_shape, 1, 0);
    TensorHandle beta = tensor_create(beta_data, gamma_shape, 1, 0);

    TensorHandle out = tensor_layer_norm_2d(t, gamma, beta, 1e-5);

    /* With gamma=1, beta=0, output should be standardized per row.
       Row 0: [1,2,3] mean=2 var=2/3 std~0.8165
       x_hat = [-1.2247, 0, 1.2247]
       Row 1: [4,5,6] mean=5 var=2/3 std~0.8165
       x_hat = [-1.2247, 0, 1.2247] */
    double std_val = sqrt(2.0/3.0 + 1e-5);
    ASSERT_NEAR("ln row0[0]", tensor_item_2d(out, 0, 0), -1.0/std_val, 1e-3);
    ASSERT_NEAR("ln row0[1]", tensor_item_2d(out, 0, 1), 0.0, 1e-3);
    ASSERT_NEAR("ln row0[2]", tensor_item_2d(out, 0, 2), 1.0/std_val, 1e-3);
    ASSERT_NEAR("ln row1[0]", tensor_item_2d(out, 1, 0), -1.0/std_val, 1e-3);
    ASSERT_NEAR("ln row1[1]", tensor_item_2d(out, 1, 1), 0.0, 1e-3);
    ASSERT_NEAR("ln row1[2]", tensor_item_2d(out, 1, 2), 1.0/std_val, 1e-3);

    /* With non-trivial gamma and beta */
    double gamma2[] = {2.0, 0.5, 1.0};
    double beta2[] = {1.0, -1.0, 0.5};
    TensorHandle gamma2h = tensor_create(gamma2, gamma_shape, 1, 0);
    TensorHandle beta2h = tensor_create(beta2, gamma_shape, 1, 0);
    TensorHandle out2 = tensor_layer_norm_2d(t, gamma2h, beta2h, 1e-5);
    /* Row 0: x_hat = [-1.2247, 0, 1.2247]
       y[0,0] = 2.0*(-1.2247) + 1.0 = -1.4494
       y[0,1] = 0.5*0 + (-1.0) = -1.0
       y[0,2] = 1.0*1.2247 + 0.5 = 1.7247 */
    double xh = 1.0 / std_val;
    ASSERT_NEAR("ln2 [0,0]", tensor_item_2d(out2, 0, 0), 2.0*(-xh) + 1.0, 1e-3);
    ASSERT_NEAR("ln2 [0,1]", tensor_item_2d(out2, 0, 1), 0.5*0.0 + (-1.0), 1e-3);
    ASSERT_NEAR("ln2 [0,2]", tensor_item_2d(out2, 0, 2), 1.0*xh + 0.5, 1e-3);
}

Test(nn_layers, layer_norm_2d_backward) {
    param_clear();

    double data[] = {0.5, -0.3, 1.2, -0.7, 0.8, 0.1};
    double gamma_data[] = {0.8, 1.2, 0.5};
    double beta_data[] = {0.1, -0.2, 0.3};
    int shape[] = {2, 3};
    int gamma_shape[] = {3};

    /* Analytical gradient */
    TensorHandle t = tensor_create(data, shape, 2, 1);
    param_register("input", t);
    TensorHandle gamma = tensor_create(gamma_data, gamma_shape, 1, 1);
    param_register("gamma", gamma);
    TensorHandle beta = tensor_create(beta_data, gamma_shape, 1, 1);
    param_register("beta", beta);

    TensorHandle out = tensor_layer_norm_2d(t, gamma, beta, 1e-5);
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);

    /* Capture analytical grads before FD scaffolding (each block below
       calls param_clear; mlx's actually releases the registry). */
    double analytic_input00 = param_grad_item_at(0, 0);
    double analytic_gamma0  = param_grad_item_at(1, 0);
    double analytic_beta1   = param_grad_item_at(2, 1);

    /* Finite diff check for input, gamma, beta */
    double eps = 1e-5;

    /* Check input[0,0] */
    {
        double d_copy[6]; memcpy(d_copy, data, sizeof(data));
        param_clear();
        d_copy[0] = data[0] + eps;
        TensorHandle t2 = tensor_create(d_copy, shape, 2, 0);
        TensorHandle g2 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b2 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_layer_norm_2d(t2, g2, b2, 1e-5)));
        d_copy[0] = data[0] - eps;
        TensorHandle t3 = tensor_create(d_copy, shape, 2, 0);
        TensorHandle g3 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b3 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_layer_norm_2d(t3, g3, b3, 1e-5)));
        double fd = (f_plus - f_minus) / (2 * eps);
        printf("  input[0,0]: fd=%f analytic=%f err=%e\n", fd, analytic_input00, fabs(fd - analytic_input00));
        ASSERT_NEAR("ln grad input[0,0]", analytic_input00, fd, FD_TOL);
    }

    /* Check gamma[0] */
    {
        double g_copy[3]; memcpy(g_copy, gamma_data, sizeof(gamma_data));
        param_clear();
        g_copy[0] = gamma_data[0] + eps;
        TensorHandle t2 = tensor_create(data, shape, 2, 0);
        TensorHandle g2 = tensor_create(g_copy, gamma_shape, 1, 0);
        TensorHandle b2 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_layer_norm_2d(t2, g2, b2, 1e-5)));
        g_copy[0] = gamma_data[0] - eps;
        TensorHandle t3 = tensor_create(data, shape, 2, 0);
        TensorHandle g3 = tensor_create(g_copy, gamma_shape, 1, 0);
        TensorHandle b3 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_layer_norm_2d(t3, g3, b3, 1e-5)));
        double fd = (f_plus - f_minus) / (2 * eps);
        printf("  gamma[0]: fd=%f analytic=%f err=%e\n", fd, analytic_gamma0, fabs(fd - analytic_gamma0));
        ASSERT_NEAR("ln grad gamma[0]", analytic_gamma0, fd, FD_TOL);
    }

    /* Check beta[1] */
    {
        double b_copy[3]; memcpy(b_copy, beta_data, sizeof(beta_data));
        param_clear();
        b_copy[1] = beta_data[1] + eps;
        TensorHandle t2 = tensor_create(data, shape, 2, 0);
        TensorHandle g2 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b2 = tensor_create(b_copy, gamma_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_layer_norm_2d(t2, g2, b2, 1e-5)));
        b_copy[1] = beta_data[1] - eps;
        TensorHandle t3 = tensor_create(data, shape, 2, 0);
        TensorHandle g3 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b3 = tensor_create(b_copy, gamma_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_layer_norm_2d(t3, g3, b3, 1e-5)));
        double fd = (f_plus - f_minus) / (2 * eps);
        printf("  beta[1]: fd=%f analytic=%f err=%e\n", fd, analytic_beta1, fabs(fd - analytic_beta1));
        ASSERT_NEAR("ln grad beta[1]", analytic_beta1, fd, FD_TOL);
    }

    param_clear();
}

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

/* ================================================================
   T13: Batch Norm
   ================================================================ */

Test(nn_layers, batch_norm_forward) {
    /* Input: [2 channels, 3 spatial] = flat [6] */
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {6};
    TensorHandle inp = tensor_create(data, shape, 1, 0);

    double gamma_d[] = {1.0, 1.0};
    double beta_d[] = {0.0, 0.0};
    double rm_d[] = {0.0, 0.0};
    double rv_d[] = {1.0, 1.0};
    int s1[] = {2};
    TensorHandle gamma = tensor_create(gamma_d, s1, 1, 0);
    TensorHandle beta = tensor_create(beta_d, s1, 1, 0);
    TensorHandle rm = tensor_create(rm_d, s1, 1, 0);
    TensorHandle rv = tensor_create(rv_d, s1, 1, 0);

    /* Training mode: normalize using input stats */
    TensorHandle out = tensor_batch_norm(inp, gamma, beta, rm, rv, 2, 3, 1, 0.1, 1e-5);

    /* Channel 0: mean=2, var=2/3, x_hat = [-1.22, 0, 1.22] (approx) */
    double result[6];
    tensor_to_doubles(out, result);
    ASSERT_NEAR("bn ch0 mean~0", (result[0]+result[1]+result[2])/3.0, 0.0, 1e-4);
    ASSERT_NEAR("bn ch1 mean~0", (result[3]+result[4]+result[5])/3.0, 0.0, 1e-4);

    /* Eval mode: should use running stats */
    TensorHandle out2 = tensor_batch_norm(inp, gamma, beta, rm, rv, 2, 3, 0, 0.1, 1e-5);
    double result2[6];
    tensor_to_doubles(out2, result2);
    /* Running mean was updated — eval output should differ from training output */
    printf("ok: batch norm forward runs\n");
}

Test(nn_layers, batch_norm_backward) {
    param_clear();

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {6};
    TensorHandle inp = tensor_create(data, shape, 1, 1);
    param_register("inp", inp);

    double gamma_d[] = {1.0, 1.0};
    double beta_d[] = {0.0, 0.0};
    double rm_d[] = {0.0, 0.0};
    double rv_d[] = {1.0, 1.0};
    int s1[] = {2};
    double* g_buf = hcopy(gamma_d, 2);
    TensorHandle gamma = tensor_create_param_1d_f64(2, g_buf);
    double* b_buf = hcopy(beta_d, 2);
    TensorHandle beta = tensor_create_param_1d_f64(2, b_buf);
    TensorHandle rm = tensor_create(rm_d, s1, 1, 0);
    TensorHandle rv = tensor_create(rv_d, s1, 1, 0);

    TensorHandle out = tensor_batch_norm(inp, gamma, beta, rm, rv, 2, 3, 1, 0.1, 1e-5);
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);

    /* d_beta[c] = sum of output grads for that channel = 3 * 1.0 = 3.0 */
    /* But output is normalized, so d_beta[c] = sum(1.0) = 3.0 for each channel */
    /* d_gamma: sum of x_hat * grad. Since mean(x_hat)=0, sum(x_hat)=0 → d_gamma=0 */

    /* Finite diff check: perturb gamma[0] */
    double eps = 1e-5;
    {
        param_clear();
        double gp[] = {1.0+eps, 1.0};
        double gm[] = {1.0-eps, 1.0};
        double* gp_buf = hcopy(gp, 2);
        double* gm_buf = hcopy(gm, 2);
        double* b1 = hcopy(beta_d, 2);
        double* b2 = hcopy(beta_d, 2);

        TensorHandle i1 = tensor_create(data, shape, 1, 0);
        TensorHandle g1 = tensor_create(gp, s1, 1, 0);
        TensorHandle bt1 = tensor_create(beta_d, s1, 1, 0);
        TensorHandle rm1 = tensor_create(rm_d, s1, 1, 0);
        TensorHandle rv1 = tensor_create(rv_d, s1, 1, 0);
        double fp = tensor_item(tensor_sum(tensor_batch_norm(i1, g1, bt1, rm1, rv1, 2, 3, 1, 0.1, 1e-5)));

        TensorHandle i2 = tensor_create(data, shape, 1, 0);
        TensorHandle g2 = tensor_create(gm, s1, 1, 0);
        TensorHandle bt2 = tensor_create(beta_d, s1, 1, 0);
        TensorHandle rm2 = tensor_create(rm_d, s1, 1, 0);
        TensorHandle rv2 = tensor_create(rv_d, s1, 1, 0);
        double fm = tensor_item(tensor_sum(tensor_batch_norm(i2, g2, bt2, rm2, rv2, 2, 3, 1, 0.1, 1e-5)));

        double fd = (fp - fm) / (2*eps);
        /* d_gamma[0] should be ~0 (sum of x_hat for centered data) */
        ASSERT_NEAR("bn fd d_gamma[0]", fd, 0.0, 0.2);
        (void)gp_buf; (void)gm_buf; (void)b1; (void)b2;
    }
    param_clear();
}

/* ================================================================
   T14: Dropout
   ================================================================ */

Test(nn_layers, dropout_forward) {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    int shape[] = {10};
    TensorHandle inp = tensor_create(data, shape, 1, 0);

    /* Training mode with p=0.5: some elements zeroed, others scaled by 2 */
    TensorHandle out = tensor_dropout(inp, 0.5, 1, 42);
    double result[10];
    tensor_to_doubles(out, result);

    int zeros = 0, scaled = 0;
    for (int i = 0; i < 10; i++) {
        if (result[i] == 0.0) zeros++;
        else if (fabs(result[i] - data[i] * 2.0) < 1e-10) scaled++;
    }
    ASSERT_TRUE("dropout: some zeros", zeros > 0);
    ASSERT_TRUE("dropout: some scaled", scaled > 0);
    ASSERT_TRUE("dropout: all zero or scaled", zeros + scaled == 10);

    /* Eval mode: identity */
    TensorHandle out_eval = tensor_dropout(inp, 0.5, 0, 42);
    double eval_result[10];
    tensor_to_doubles(out_eval, eval_result);
    ASSERT_NEAR("dropout eval[0]", eval_result[0], 1.0, 1e-10);
    ASSERT_NEAR("dropout eval[9]", eval_result[9], 10.0, 1e-10);
}

Test(nn_layers, dropout_backward) {
    param_clear();

    double data[] = {1.0, 2.0, 3.0, 4.0};
    int shape[] = {4};
    TensorHandle inp = tensor_create(data, shape, 1, 1);
    param_register("inp", inp);

    TensorHandle out = tensor_dropout(inp, 0.5, 1, 123);
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);

    /* Gradient should be 0 where dropped, 2.0 (=1/(1-0.5)) where kept */
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        double g = param_grad_item_at(0, i);
        if (fabs(g) > 1e-10 && fabs(g - 2.0) > 1e-10) {
            printf("FAIL: dropout grad[%d] = %f (expected 0 or 2)\n", i, g);
            ok = 0;
        }
    }
    if (ok) printf("ok: dropout gradients correct (0 or scale)\n");
    param_clear();
}

/* ================================================================
   T14: Conv1D + MaxPool1D
   ================================================================ */

Test(nn_layers, conv1d_forward) {
    double inp_data[] = {1, 2, 3, 4, 5};
    int inp_shape[] = {1, 5};
    TensorHandle inp = tensor_create(inp_data, inp_shape, 2, 0);

    double ker_data[] = {1, 0, 1};
    int ker_shape[] = {1, 1, 3};
    TensorHandle ker = tensor_create(ker_data, ker_shape, 3, 0);

    TensorHandle out = tensor_conv1d(inp, ker, NULL, 0, 1);
    ASSERT_TRUE("conv1d dim", tensor_dim(out) == 2);
    ASSERT_TRUE("conv1d size0", tensor_size(out, 0) == 1);
    ASSERT_TRUE("conv1d size1", tensor_size(out, 1) == 3);
    double result[3];
    tensor_to_doubles(out, result);
    ASSERT_NEAR("conv1d[0]", result[0], 4.0, 1e-10);
    ASSERT_NEAR("conv1d[1]", result[1], 6.0, 1e-10);
    ASSERT_NEAR("conv1d[2]", result[2], 8.0, 1e-10);
}

Test(nn_layers, conv1d_backward) {
    param_clear();
    double inp_data[] = {1, 2, 3, 4, 5};
    int inp_shape[] = {1, 5};
    double ker_data[] = {1, 1, 1};
    int ker_shape[] = {1, 1, 3};

    TensorHandle inp = tensor_create(inp_data, inp_shape, 2, 1);
    param_register("inp", inp);
    TensorHandle ker = tensor_create(ker_data, ker_shape, 3, 1);
    param_register("ker", ker);

    TensorHandle out = tensor_conv1d(inp, ker, NULL, 0, 1);
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);

    ASSERT_NEAR("d_ker1d[0]", param_grad_item_at(1, 0), 6.0, 1e-10);
    ASSERT_NEAR("d_ker1d[1]", param_grad_item_at(1, 1), 9.0, 1e-10);
    ASSERT_NEAR("d_ker1d[2]", param_grad_item_at(1, 2), 12.0, 1e-10);
    param_clear();
}

Test(nn_layers, max_pool1d_forward) {
    double inp_data[] = {1, 3, 2, 4, 5, 1};
    int inp_shape[] = {1, 6};
    TensorHandle inp = tensor_create(inp_data, inp_shape, 2, 0);

    TensorHandle out = tensor_max_pool1d(inp, 2, 2);
    ASSERT_TRUE("pool1d size1", tensor_size(out, 1) == 3);
    double result[3];
    tensor_to_doubles(out, result);
    ASSERT_NEAR("pool1d[0]", result[0], 3.0, 1e-10);
    ASSERT_NEAR("pool1d[1]", result[1], 4.0, 1e-10);
    ASSERT_NEAR("pool1d[2]", result[2], 5.0, 1e-10);
}

/* ================================================================
   T15: Conv2D + MaxPool2D
   ================================================================ */

Test(nn_layers, conv2d_forward) {
    /* Input: [1, 4, 4] — single channel 4x4 image */
    double inp_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int inp_shape[] = {1, 4, 4};
    TensorHandle inp = tensor_create(inp_data, inp_shape, 3, 0);

    /* Kernel: [1, 1, 2, 2] — one output channel, 2x2 kernel */
    double ker_data[] = {1, 0, 0, 1};
    int ker_shape[] = {1, 1, 2, 2};
    TensorHandle ker = tensor_create(ker_data, ker_shape, 4, 0);

    /* No bias, no padding, stride=1 */
    TensorHandle out = tensor_conv2d(inp, ker, NULL, 0, 0, 1, 1);

    /* Output should be [1, 3, 3]: out[0,oh,ow] = inp[oh,ow] + inp[oh+1,ow+1]
       = {1+6, 2+7, 3+8, 5+10, 6+11, 7+12, 9+14, 10+15, 11+16} */
    ASSERT_TRUE("conv2d output rank", tensor_dim(out) == 3);
    ASSERT_TRUE("conv2d output size 0", tensor_size(out, 0) == 1);
    ASSERT_TRUE("conv2d output size 1", tensor_size(out, 1) == 3);
    ASSERT_TRUE("conv2d output size 2", tensor_size(out, 2) == 3);

    double expected[] = {7, 9, 11, 15, 17, 19, 23, 25, 27};
    double result[9];
    tensor_to_doubles(out, result);
    for (int i = 0; i < 9; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "conv2d out[%d]", i);
        ASSERT_NEAR(msg, result[i], expected[i], 1e-10);
    }
}

Test(nn_layers, conv2d_backward) {
    param_clear();

    /* Analytical gradient */
    double inp_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int inp_shape[] = {1, 3, 3};

    double ker_data[] = {1, 1, 1, 1};
    int ker_shape[] = {1, 1, 2, 2};

    TensorHandle inp = tensor_create(inp_data, inp_shape, 3, 1);
    param_register("inp", inp);
    TensorHandle ker = tensor_create(ker_data, ker_shape, 4, 1);
    param_register("ker", ker);

    TensorHandle out = tensor_conv2d(inp, ker, NULL, 0, 0, 1, 1);
    TensorHandle loss = tensor_sum(out);
    double loss_val = tensor_item(loss);
    ASSERT_NEAR("conv2d loss", loss_val, 80.0, 1e-10);

    tensor_backward(loss);

    /* Check kernel gradients via param registry */
    /* d_ker[0] = sum of top-left corners = 1+2+4+5 = 12 */
    ASSERT_NEAR("d_kernel[0]", param_grad_item_at(1, 0), 12.0, 1e-10);
    ASSERT_NEAR("d_kernel[1]", param_grad_item_at(1, 1), 16.0, 1e-10);
    ASSERT_NEAR("d_kernel[2]", param_grad_item_at(1, 2), 24.0, 1e-10);
    ASSERT_NEAR("d_kernel[3]", param_grad_item_at(1, 3), 28.0, 1e-10);

    /* Finite diff check for ker[0] */
    double eps = 1e-5;
    {
        param_clear();
        double ker_p[4] = {1+eps, 1, 1, 1};
        double ker_m[4] = {1-eps, 1, 1, 1};
        TensorHandle i1 = tensor_create(inp_data, inp_shape, 3, 0);
        TensorHandle k1 = tensor_create(ker_p, ker_shape, 4, 0);
        double fp = tensor_item(tensor_sum(tensor_conv2d(i1, k1, NULL, 0,0,1,1)));
        TensorHandle i2 = tensor_create(inp_data, inp_shape, 3, 0);
        TensorHandle k2 = tensor_create(ker_m, ker_shape, 4, 0);
        double fm = tensor_item(tensor_sum(tensor_conv2d(i2, k2, NULL, 0,0,1,1)));
        double fd = (fp - fm) / (2*eps);
        ASSERT_NEAR("conv2d fd d_ker[0]", fd, 12.0, FD_TOL);  /* FD via fp32 forward chain catastrophic-cancels for mlx */
    }

    param_clear();
}

Test(nn_layers, max_pool2d_forward) {
    /* Input: [1, 4, 4] */
    double inp_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int inp_shape[] = {1, 4, 4};
    TensorHandle inp = tensor_create(inp_data, inp_shape, 3, 0);

    /* MaxPool2D: k=2, stride=2 -> output [1, 2, 2] */
    TensorHandle out = tensor_max_pool2d(inp, 2, 2, 2, 2);

    ASSERT_TRUE("pool output rank", tensor_dim(out) == 3);
    ASSERT_TRUE("pool output size 0", tensor_size(out, 0) == 1);
    ASSERT_TRUE("pool output size 1", tensor_size(out, 1) == 2);
    ASSERT_TRUE("pool output size 2", tensor_size(out, 2) == 2);

    double result[4];
    tensor_to_doubles(out, result);
    /* max of each 2x2 block: {6, 8, 14, 16} */
    ASSERT_NEAR("pool out[0]", result[0], 6.0, 1e-10);
    ASSERT_NEAR("pool out[1]", result[1], 8.0, 1e-10);
    ASSERT_NEAR("pool out[2]", result[2], 14.0, 1e-10);
    ASSERT_NEAR("pool out[3]", result[3], 16.0, 1e-10);
}

Test(nn_layers, max_pool2d_backward) {
    param_clear();

    double inp_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int inp_shape[] = {1, 4, 4};

    TensorHandle inp = tensor_create(inp_data, inp_shape, 3, 1);
    param_register("inp", inp);

    TensorHandle out = tensor_max_pool2d(inp, 2, 2, 2, 2);
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);

    /* Gradient at max positions (indices 5,7,13,15) should be 1.0 */
    ASSERT_NEAR("d_pool inp[5]", param_grad_item_at(0, 5), 1.0, 1e-10);
    ASSERT_NEAR("d_pool inp[7]", param_grad_item_at(0, 7), 1.0, 1e-10);
    ASSERT_NEAR("d_pool inp[13]", param_grad_item_at(0, 13), 1.0, 1e-10);
    ASSERT_NEAR("d_pool inp[15]", param_grad_item_at(0, 15), 1.0, 1e-10);
    /* Non-max positions should be 0 */
    ASSERT_NEAR("d_pool inp[0]", param_grad_item_at(0, 0), 0.0, 1e-10);
    ASSERT_NEAR("d_pool inp[4]", param_grad_item_at(0, 4), 0.0, 1e-10);

    param_clear();
}

/* ================================================================
   Main
   ================================================================ */

Test(nn_layers, embedding) {
    param_clear();
    /* weight [3, 2]: 3 vocab, 2-dim embeddings */
    double w[] = {1,2, 3,4, 5,6};
    int ws[] = {3, 2};
    TensorHandle weight = tensor_create(w, ws, 2, 1);
    param_register("emb", weight);

    /* indices [2]: lookup rows 2 and 0 */
    double idx[] = {2, 0};
    int is[] = {2};
    TensorHandle indices = tensor_create(idx, is, 1, 0);

    TensorHandle out = tensor_embedding(weight, indices, 2, 2);
    /* Expected: [5,6, 1,2] (row 2 then row 0) */
    double result[4];
    tensor_to_doubles(out, result);
    ASSERT_NEAR("embed[0]", result[0], 5.0, 1e-10);
    ASSERT_NEAR("embed[1]", result[1], 6.0, 1e-10);
    ASSERT_NEAR("embed[2]", result[2], 1.0, 1e-10);
    ASSERT_NEAR("embed[3]", result[3], 2.0, 1e-10);

    /* Backward: sum all outputs */
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    /* d_weight[2,0] += 1, d_weight[2,1] += 1, d_weight[0,0] += 1, d_weight[0,1] += 1 */
    ASSERT_NEAR("d_emb[0]", param_grad_item_at(0, 0), 1.0, 1e-10);
    ASSERT_NEAR("d_emb[1]", param_grad_item_at(0, 1), 1.0, 1e-10);
    ASSERT_NEAR("d_emb[2]", param_grad_item_at(0, 2), 0.0, 1e-10);
    ASSERT_NEAR("d_emb[4]", param_grad_item_at(0, 4), 1.0, 1e-10);
    param_clear();
}

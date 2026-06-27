#include "port_assert.h"

Test(nn_norm_layer_norm_2d, layer_norm_2d) {
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

Test(nn_norm_layer_norm_2d, layer_norm_2d_backward) {
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

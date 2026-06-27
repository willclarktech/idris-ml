#include "port_assert.h"

Test(linear_linalg_bmm, bmm_forward) {
    /* a = [2, 2, 3], b = [3, 2] => result = [2, 2, 2] */
    double a_data[] = {
        /* batch 0: [[1,2,3],[4,5,6]] */
        1, 2, 3, 4, 5, 6,
        /* batch 1: [[7,8,9],[10,11,12]] */
        7, 8, 9, 10, 11, 12
    };
    double b_data[] = {1, 0, 0, 1, 1, 1};  /* [[1,0],[0,1],[1,1]] */
    int a_shape[] = {2, 2, 3};
    int b_shape[] = {3, 2};

    TensorHandle a = tensor_create(a_data, a_shape, 3, 0);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 0);
    TensorHandle c = tensor_bmm(a, b);

    /* Read results into flat buffer for rank-3 indexing */
    double out[8];
    tensor_to_doubles(c, out);
    /* batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]] */
    ASSERT_NEAR("bmm[0,0,0]", out[0], 4.0, 1e-10);
    ASSERT_NEAR("bmm[0,0,1]", out[1], 5.0, 1e-10);
    ASSERT_NEAR("bmm[0,1,0]", out[2], 10.0, 1e-10);
    ASSERT_NEAR("bmm[0,1,1]", out[3], 11.0, 1e-10);
    /* batch 1: [[7,8,9],[10,11,12]] @ [[1,0],[0,1],[1,1]] = [[16,17],[22,23]] */
    ASSERT_NEAR("bmm[1,0,0]", out[4], 16.0, 1e-10);
    ASSERT_NEAR("bmm[1,0,1]", out[5], 17.0, 1e-10);
    ASSERT_NEAR("bmm[1,1,0]", out[6], 22.0, 1e-10);
    ASSERT_NEAR("bmm[1,1,1]", out[7], 23.0, 1e-10);
}

Test(linear_linalg_bmm, bmm_backward) {
    param_clear();

    double a_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    double b_data[] = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    int a_shape[] = {2, 2, 3};
    int b_shape[] = {3, 2};

    TensorHandle a = tensor_create(a_data, a_shape, 3, 1);
    param_register("a", a);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 1);
    param_register("b", b);

    TensorHandle c = tensor_bmm(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);

    /* Capture analytical grads before any param_clear (FD blocks below
       call param_clear; mlx's actually releases the registry). */
    double analytic_a0 = param_grad_item_at(0, 0);
    double analytic_b0 = param_grad_item_at(1, 0);

    /* Finite diff check for a[0] (first element of batch 0) */
    double eps = 1e-5;
    {
        double a_copy[12]; memcpy(a_copy, a_data, sizeof(a_data));
        param_clear();
        a_copy[0] = a_data[0] + eps;
        TensorHandle a2 = tensor_create(a_copy, a_shape, 3, 0);
        TensorHandle b2 = tensor_create(b_data, b_shape, 2, 0);
        double f_plus = tensor_item(tensor_sum(tensor_bmm(a2, b2)));
        a_copy[0] = a_data[0] - eps;
        TensorHandle a3 = tensor_create(a_copy, a_shape, 3, 0);
        TensorHandle b3 = tensor_create(b_data, b_shape, 2, 0);
        double f_minus = tensor_item(tensor_sum(tensor_bmm(a3, b3)));
        double fd = (f_plus - f_minus) / (2 * eps);
        printf("  a[0]: fd=%f analytic=%f err=%e\n", fd, analytic_a0, fabs(fd - analytic_a0));
        ASSERT_NEAR("bmm grad a[0]", analytic_a0, fd, FD_TOL);
    }

    /* Finite diff check for b[0] (weight grad, accumulated across batch) */
    {
        double b_copy[6]; memcpy(b_copy, b_data, sizeof(b_data));
        param_clear();
        b_copy[0] = b_data[0] + eps;
        TensorHandle a2 = tensor_create(a_data, a_shape, 3, 0);
        TensorHandle b2 = tensor_create(b_copy, b_shape, 2, 0);
        double f_plus = tensor_item(tensor_sum(tensor_bmm(a2, b2)));
        b_copy[0] = b_data[0] - eps;
        TensorHandle a3 = tensor_create(a_data, a_shape, 3, 0);
        TensorHandle b3 = tensor_create(b_copy, b_shape, 2, 0);
        double f_minus = tensor_item(tensor_sum(tensor_bmm(a3, b3)));
        double fd = (f_plus - f_minus) / (2 * eps);
        printf("  b[0]: fd=%f analytic=%f err=%e\n", fd, analytic_b0, fabs(fd - analytic_b0));
        ASSERT_NEAR("bmm grad b[0]", analytic_b0, fd, FD_TOL);
    }

    param_clear();
}

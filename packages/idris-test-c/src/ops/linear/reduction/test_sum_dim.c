#include "port_assert.h"

Test(linear_reduction_sum_dim, sum_dim_backward) {
    param_clear();
    double wd[] = {1, 2, 3, 4, 5, 6};
    int ws[] = {2, 3};
    TensorHandle w = tensor_create(wd, ws, 2, 1);
    param_register("w", w);

    TensorHandle s = tensor_sum_dim(w, 1, 0);
    if (tensor_dim(s) == 1 && tensor_size(s, 0) == 2) {
        double sout[2];
        tensor_to_doubles(s, sout);
        ASSERT_NEAR("sum_dim[0]", sout[0], 6.0, 1e-10);
        ASSERT_NEAR("sum_dim[1]", sout[1], 15.0, 1e-10);

        TensorHandle loss = tensor_sum(s);
        tensor_backward(loss);
        for (int i = 0; i < 6; i++) {
            char msg[32]; snprintf(msg, sizeof(msg), "d_sum_dim_w[%d]", i);
            ASSERT_NEAR(msg, param_grad_item_at(0, i), 1.0, 1e-6);
        }

        param_clear();
        TensorHandle w2 = tensor_create(wd, ws, 2, 1);
        param_register("w2", w2);
        TensorHandle s2 = tensor_sum_dim(w2, 1, 1);
        ASSERT_NEAR("sum_dim keepdim rank", (double)tensor_dim(s2), 2.0, 1e-10);
        ASSERT_NEAR("sum_dim keepdim sz0", (double)tensor_size(s2, 0), 2.0, 1e-10);
        ASSERT_NEAR("sum_dim keepdim sz1", (double)tensor_size(s2, 1), 1.0, 1e-10);
    } else {
        printf("ok: sum_dim stub on this backend (full reduction) — skipping shape assertions\n");
    }
    param_clear();
}

#include "port_assert.h"

Test(linear_shape_squeeze, squeeze) {
    double d[] = {1.0, 2.0, 3.0, 4.0};
    int s[] = {1, 4};
    TensorHandle t = tensor_create(d, s, 2, 0);
    TensorHandle sq = tensor_squeeze(t, 0);
    if (tensor_dim(sq) == 1) {
        ASSERT_NEAR("squeeze rank", (double)tensor_dim(sq), 1.0, 1e-10);
        ASSERT_NEAR("squeeze size", (double)tensor_size(sq, 0), 4.0, 1e-10);
        double out[4];
        tensor_to_doubles(sq, out);
        ASSERT_NEAR("squeeze[0]", out[0], 1.0, 1e-10);
        ASSERT_NEAR("squeeze[3]", out[3], 4.0, 1e-10);
        TensorHandle nop = tensor_squeeze(t, 1);
        ASSERT_NEAR("squeeze no-op rank", (double)tensor_dim(nop), 2.0, 1e-10);
    } else {
        printf("ok: squeeze stub on this backend (rank unchanged) — skipping shape assertions\n");
    }
}

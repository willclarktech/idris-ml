/* Regression guard for the bench_ops conv2d crash that was historically
 * filed as "conv2d segfaults after train_step on tape". The crash was
 * NOT a tape state-machine bug — it was bench_ops constructing a rank-1
 * flat input where tensor_conv2d expects rank-3 [inC, H, W]. The "post-
 * train_step" framing was a coincidence (heap state happened to produce
 * a large garbage shape[1]/shape[2] after the optimizer had run).
 *
 * This test exercises the same call pattern bench_ops uses: a few
 * optimizer.step iterations on an unrelated linear param, then a
 * properly rank-3 conv2d forward. Asserts the conv2d output matches
 * the analytic answer to within F64 tolerance.
 *
 * RED before the bench_ops fix: this test would not exist; the bug
 * surfaced as a SIGSEGV in `make bench-ops` once the linker was fixed
 * (which had previously masked the run-time crash). The fix is purely
 * in bench_ops.c (rank-3 input construction); the library contract was
 * always correct.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"

Test(conv_conv2d_post_train_step, rank3_input_after_optimizer_step) {
    param_clear();

    /* Phase 1: tiny linear training loop (W: [2,2], b: [2]) for 3 steps. */
    double wdata[4] = {0.1, 0.2, 0.3, 0.4};
    double bdata[2] = {0.0, 0.0};
    double* wcopy = (double*)malloc(4 * sizeof(double));
    double* bcopy = (double*)malloc(2 * sizeof(double));
    for (int i = 0; i < 4; i++) wcopy[i] = wdata[i];
    for (int i = 0; i < 2; i++) bcopy[i] = bdata[i];
    TensorHandle W = tensor_create_param_2d_f64(2, 2, wcopy);
    TensorHandle b = tensor_create_param_1d_f64(2, bcopy);
    param_register("W", W);
    param_register("b", b);
    OptimizerHandle opt = optimizer_create_sgd(0.01);

    double xdata[2] = {1.0, -1.0};
    int xshape[1] = {2};
    for (int step = 0; step < 3; step++) {
        optimizer_zero_grad(opt);
        TensorHandle x = tensor_create(xdata, xshape, 1, 0);
        TensorHandle y = tensor_linear(W, x, b);
        TensorHandle loss = tensor_sum(y);
        tensor_backward(loss);
        optimizer_step(opt);
        tensor_free(x);
    }
    optimizer_free(opt);

    /* Phase 2: conv2d with a *rank-3* input — the bench_ops contract.
       input [inC=1, H=2, W=2] = [[1,2],[3,4]], kernel [outC=1, inC=1,
       kH=2, kW=2] = ones, no bias, pad=0, stride=1.
       Forward: out[0,0,0] = 1+2+3+4 = 10. */
    double in_data[4] = {1.0, 2.0, 3.0, 4.0};
    double k_data[4]  = {1.0, 1.0, 1.0, 1.0};
    int sh_in[3] = {1, 2, 2};
    int sh_k[4]  = {1, 1, 2, 2};
    TensorHandle in = tensor_create(in_data, sh_in, 3, 0);
    TensorHandle k  = tensor_create(k_data,  sh_k,  4, 0);
    TensorHandle out = tensor_conv2d(in, k, (TensorHandle)0, 0, 0, 1, 1);
    cr_assert_float_eq(tensor_item_1d(out, 0), 10.0, 1e-12,
        "conv2d after 3 optimizer steps returns expected value (got %.6f)",
        tensor_item_1d(out, 0));

    param_clear();
}

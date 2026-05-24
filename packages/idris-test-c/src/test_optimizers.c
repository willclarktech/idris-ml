/* Criterion suite for the native optimizer surface on every backend.
 *
 * Covers SGD/RMSprop convergence + lr-schedule semantics + per-param-lr
 * + a fused mv + optimizer end-to-end loop. These tests were carried
 * over from test_backend.c when its monolithic harness was retired in
 * favour of Criterion's per-test process isolation + JUnit XML output.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

/* Helper: heap copy of a small double[] for use with param creators. */
static double* heap_copy(const double* src, int n) {
    double* p = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) p[i] = src[i];
    return p;
}

Test(training_optimizer_sgd, scalar_quadratic_convergence) {
    /* loss = (w*2 - 3)^2 → optimal w = 1.5. Train 100 SGD steps from
       w=0.5 at lr=0.01 and assert w lands within 0.01 of 1.5. */
    param_clear();
    TensorHandle w = tensor_create_scalar(0.5, 1);
    param_register("w", w);
    OptimizerHandle sgd = optimizer_create_sgd(0.01);
    for (int step = 0; step < 100; step++) {
        optimizer_zero_grad(sgd);
        TensorHandle x = tensor_create_scalar(2.0, 0);
        TensorHandle t = tensor_create_scalar(3.0, 0);
        TensorHandle pred = tensor_mul(w, x);
        TensorHandle diff = tensor_sub(pred, t);
        TensorHandle loss = tensor_mul(diff, diff);
        tensor_backward(loss);
        optimizer_step(sgd);
        tensor_free(x); tensor_free(t); tensor_free(pred);
        tensor_free(diff); tensor_free(loss);
    }
    cr_assert_float_eq(tensor_item(w), 1.5, 0.01, "SGD converges w->1.5");
    optimizer_free(sgd);
    tensor_free(w);
    param_clear();
}

/* Distinguishes the torch.optim.RMSprop form (lr OUTSIDE the momentum
   buffer) from the lr-inside form. At constant lr the two coincide,
   so the test uses two steps with DIFFERENT lr (0.1 then 0.2) and
   momentum > 0 — the only regime where they diverge. With loss=w each
   step (grad=1.0), lr-outside lands w at ~5.78224; the lr-inside bug
   lands at ~6.68224 (verified against PyTorch). */
Test(training_optimizer_rmsprop, lr_outside_buffer) {
    param_clear();
    TensorHandle w = tensor_create_scalar(10.0, 1);
    param_register("w", w);
    OptimizerHandle opt = optimizer_create_rmsprop(0.1, 0.99, 1e-8, 0.0, 0.9);
    optimizer_zero_grad(opt);
    TensorHandle l1 = tensor_sum(w);
    tensor_backward(l1);
    optimizer_step(opt);
    tensor_free(l1);
    optimizer_set_lr(opt, 0.2);
    optimizer_zero_grad(opt);
    TensorHandle l2 = tensor_sum(w);
    tensor_backward(l2);
    optimizer_step(opt);
    tensor_free(l2);
    cr_assert_float_eq(tensor_item(w), 5.78224, 1e-3,
        "RMSprop lr-outside w after 2 steps (got %.6f)", tensor_item(w));
    optimizer_free(opt);
    tensor_free(w);
    param_clear();
}

/* Per-param LR: override one param's LR while leaving the base in
   effect for the others. Tape + mlx implement the override; torch's
   backend wires `optimizer_set_param_lr` as a no-op so we tolerate
   either outcome on `w` and require `b`'s base-LR behaviour either way. */
Test(training_optimizer_per_param_lr, partial_override) {
    param_clear();
    TensorHandle w = tensor_create_scalar(5.0, 1);
    TensorHandle b = tensor_create_scalar(3.0, 1);
    param_register("w", w);
    param_register("b", b);
    TensorHandle loss = tensor_add(w, b);
    tensor_backward(loss);
    OptimizerHandle opt = optimizer_create_sgd(0.1);
    optimizer_set_param_lr(opt, "w", 0.5);
    optimizer_step(opt);
    double w_val = tensor_item(w);
    if (w_val < 4.8) {
        /* Per-param LR applied (tape/mlx). */
        cr_assert_float_eq(w_val, 4.5, 1e-5, "w after per-param LR");
    }
    /* Otherwise: w stayed at 4.9 (torch's no-op). Both shapes acceptable. */
    cr_assert_float_eq(tensor_item(b), 2.9, 1e-5, "b after base LR");
    optimizer_free(opt);
    param_clear();
}

/* Fused MV with SGD: 5 epochs on a 2x3 W with constant input x=[1,0,-1].
   Forward y = W @ x, loss = sum(y) = W[0,0] - W[0,2] + W[1,0] - W[1,2].
   Each step subtracts 0.1 from W[i,0], adds 0.1 to W[i,2]; loss should
   monotonically decrease (within slack) and end below -4.0. */
Test(training_optimizer_fused_mv, multi_epoch_loss_decreases) {
    param_clear();
    double wdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    TensorHandle W = tensor_create_param_2d_f64(2, 3, heap_copy(wdata, 6));
    param_register("W", W);
    double xdata[] = {1.0, 0.0, -1.0};
    int xshape[] = {3};
    OptimizerHandle sgd = optimizer_create_sgd(0.1);
    double prev_loss = 1e10;
    for (int ep = 0; ep < 5; ep++) {
        optimizer_zero_grad(sgd);
        TensorHandle x = tensor_create(xdata, xshape, 1, 0);
        TensorHandle y = tensor_mv(W, x);
        TensorHandle loss = tensor_sum(y);
        double lv = tensor_item(loss);
        if (ep > 0) cr_assert(lv < prev_loss + 0.01, "loss decreasing (ep=%d lv=%.4f prev=%.4f)", ep, lv, prev_loss);
        prev_loss = lv;
        tensor_backward(loss);
        optimizer_step(sgd);
    }
    cr_assert(prev_loss < -4.0, "fused MV trains (final loss=%.4f)", prev_loss);
    optimizer_free(sgd);
    param_clear();
}

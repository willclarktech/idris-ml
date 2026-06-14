/* Criterion suite for the native optimizer surface on every backend.
 *
 * Covers SGD/RMSprop convergence + lr-schedule semantics + per-param-lr
 * + a fused mv + optimizer end-to-end loop. These tests were carried
 * over from test_backend.c when its monolithic harness was retired in
 * favour of Criterion's per-test process isolation + JUnit XML output.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

/* Helper: heap copy of a small double[] for use with param creators. */
static double* heap_copy(const double* src, int n) {
    double* p = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) p[i] = src[i];
    return p;
}

/* Fused entry point: zero_grad + backward + clip + step in one call —
   the path Idris's nativeTrainStep drives. The unfused sequence above
   (scalar_quadratic_convergence) passed on every backend while the
   Idris suite segfaulted inside this entry on torch, so the fused
   path needs its own coverage. loss = w*w from w=1 at lr=0.1: one
   step gives w = 1 - 0.1*2 = 0.8. */
#ifdef BACKEND_TORCH
/* Lifecycle contract behind the 2026-06-12 use-after-free: torch's
   create-then-migrate param path must use the PERSISTENT to_device
   variant. The tracked tensor_to_device pushes its result onto the
   intermediates vector, so the first optimizer_step's
   free_intermediates() deleted params created through it — every
   later read (tensor_item, registry walks, saves) was a
   use-after-free (the Idris Optimizer-suite segfault and the
   Hpo.LrFinder SIGABRT class). */
extern int tensor_live_count(void);
extern TensorHandle tensor_to_device(TensorHandle t, const char* device);
extern TensorHandle tensor_to_device_persistent(TensorHandle t, const char* device);

Test(training_optimizer_sgd, to_device_persistent_param_survives_steps) {
    param_clear();
    TensorHandle w0 = tensor_create_scalar(1.0, 1);
    int live0 = tensor_live_count();
    TensorHandle tracked = tensor_to_device(w0, "cpu");
    cr_assert_eq(tensor_live_count(), live0 + 1,
                 "tracked to_device is intermediates-tracked (contrast case)");
    (void)tracked;
    TensorHandle w = tensor_to_device_persistent(w0, "cpu");
    cr_assert_eq(tensor_live_count(), live0 + 1,
                 "persistent to_device must NOT be intermediates-tracked");
    param_register("tdp_w", w);
    TensorHandle loss = tensor_mul(w, w);
    OptimizerHandle opt = optimizer_create_sgd(0.1);
    native_train_step(opt, 0, 0.0, loss, 1.0);
    /* free_intermediates ran inside the step; the migrated param must
       still be readable — this read was the Idris crash site. */
    double w1 = tensor_item(w);
    cr_assert_float_eq(w1, 0.8, 1e-12, "param survives first fused step (got %.15f)", w1);
    TensorHandle loss2 = tensor_mul(w, w);
    native_train_step(opt, 0, 0.0, loss2, tensor_item(loss2));
    double w2 = tensor_item(w);
    cr_assert_float_eq(w2, 0.64, 1e-12, "param survives second fused step (got %.15f)", w2);
    optimizer_free(opt);
    param_clear();
}
#endif /* BACKEND_TORCH */

Test(training_optimizer_sgd, native_train_step_fused) {
    param_clear();
    TensorHandle w = tensor_create_scalar(1.0, 1);
    param_register("nts_w", w);
    TensorHandle loss = tensor_mul(w, w);
    OptimizerHandle opt = optimizer_create_sgd(0.1);
    double lv = native_train_step(opt, /*clip_mode=*/0, /*clip_val=*/0.0, loss, 1.0);
    cr_assert_float_eq(lv, 1.0, 1e-12, "fused step returns loss_val (got %f)", lv);
    /* 1e-6 tolerance: mlx's legacy tensor_create_scalar routes to F32
       (see backend.h), so the step lands at F32 precision there. */
    double w1 = tensor_item(w);
    cr_assert_float_eq(w1, 0.8, 1e-6, "w after one fused sgd step: got %.15f", w1);
    optimizer_free(opt);
    param_clear();
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
   effect for the others. Implemented on all three backends (tape + mlx
   per-param arrays; torch via per-param LR buckets in optimizer_step).
   w override 0.5 → 5.0 - 0.5*1 = 4.5; b base 0.1 → 3.0 - 0.1*1 = 2.9. */
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
    cr_assert_float_eq(w_val, 4.5, 1e-5, "w after per-param LR (got %.6f)", w_val);
    cr_assert_float_eq(tensor_item(b), 2.9, 1e-5, "b after base LR (got %.6f)", tensor_item(b));
    optimizer_free(opt);
    param_clear();
}

/* AdamW foreach (BLAS-1 moment update) vs scalar inner loop. The BLAS path
   in `adamw_foreach_param` uses Accelerate's cblas_dscal + cblas_daxpy
   for the m-update on F64 params ≥ 256 elements; both other components
   (v-update and weight update) stay scalar and bit-match the scalar path.
   FMA inside cblas_daxpy can introduce ULP-level drift in the m buffer
   that propagates through the bias-correct + sqrt-div; over 50 AdamW
   steps with realistic hyperparameters the cumulative drift stays well
   under 1e-12. Same param shape on both phases hits the n ≥ 256 BLAS
   gate so both code paths exercise BLAS in the moment update.

   Foreach is the default; this test uses TAPE_OPTIMIZER_FOREACH=0 as an
   internal opt-out to force scalar for phase 1. */
Test(training_optimizer_adamw_foreach, matches_scalar_on_256_elem_param) {
    const int N = 256;
    const int NSTEPS = 50;
    double scalar_w[256], foreach_w[256];

    /* Phase 1: scalar path (env opt-out). */
    setenv("TAPE_OPTIMIZER_FOREACH", "0", 1);
    param_clear();
    double wdata1[256];
    for (int i = 0; i < N; i++) wdata1[i] = (double)(i % 7) * 0.1 - 0.3;
    TensorHandle W1 = tensor_create_param_2d_f64(16, 16, heap_copy(wdata1, N));
    param_register("W", W1);
    OptimizerHandle opt1 = optimizer_create_adamw(0.01, 0.9, 0.999, 1e-8, 0.01);
    for (int step = 0; step < NSTEPS; step++) {
        optimizer_zero_grad(opt1);
        TensorHandle s = tensor_sum(W1);
        tensor_backward(s);
        optimizer_step(opt1);
        tensor_free(s);
    }
    tensor_to_doubles(W1, scalar_w);
    optimizer_free(opt1);
    param_clear();

    /* Phase 2: foreach path (default; env unset). Identical init + grad
       trajectory. */
    unsetenv("TAPE_OPTIMIZER_FOREACH");
    double wdata2[256];
    for (int i = 0; i < N; i++) wdata2[i] = (double)(i % 7) * 0.1 - 0.3;
    TensorHandle W2 = tensor_create_param_2d_f64(16, 16, heap_copy(wdata2, N));
    param_register("W", W2);
    OptimizerHandle opt2 = optimizer_create_adamw(0.01, 0.9, 0.999, 1e-8, 0.01);
    for (int step = 0; step < NSTEPS; step++) {
        optimizer_zero_grad(opt2);
        TensorHandle s = tensor_sum(W2);
        tensor_backward(s);
        optimizer_step(opt2);
        tensor_free(s);
    }
    tensor_to_doubles(W2, foreach_w);
    optimizer_free(opt2);
    param_clear();

    /* Compare element-wise. */
    double max_diff = 0.0;
    int max_idx = -1;
    for (int i = 0; i < N; i++) {
        double d = fabs(scalar_w[i] - foreach_w[i]);
        if (d > max_diff) { max_diff = d; max_idx = i; }
    }
    cr_assert(max_diff < 1e-12,
        "AdamW foreach diverged from scalar: max_diff=%.6e at idx=%d (scalar=%.12g foreach=%.12g)",
        max_diff, max_idx, scalar_w[max_idx], foreach_w[max_idx]);
}

/* Adam (not AdamW) foreach vs scalar. Same shape as the AdamW pair above,
   but using `optimizer_create_adam` (no weight-decay arg). Adam reduces to
   AdamW with wd=0; the `calloc` in `tape_optimizer_create_adam` ensures
   `opt->weight_decay == 0`, so the existing foreach math (wd term self-
   zeroes) is correct for Adam without any flag plumbing. This test
   exercises the gate-widening from `opt->type == 3` to `opt->type == 2 ||
   opt->type == 3`. */
Test(training_optimizer_adam_foreach, matches_scalar_on_256_elem_param) {
    const int N = 256;
    const int NSTEPS = 50;
    double scalar_w[256], foreach_w[256];

    /* Phase 1: scalar path (env opt-out). */
    setenv("TAPE_OPTIMIZER_FOREACH", "0", 1);
    param_clear();
    double wdata1[256];
    for (int i = 0; i < N; i++) wdata1[i] = (double)(i % 7) * 0.1 - 0.3;
    TensorHandle W1 = tensor_create_param_2d_f64(16, 16, heap_copy(wdata1, N));
    param_register("W", W1);
    OptimizerHandle opt1 = optimizer_create_adam(0.01, 0.9, 0.999, 1e-8);
    for (int step = 0; step < NSTEPS; step++) {
        optimizer_zero_grad(opt1);
        TensorHandle s = tensor_sum(W1);
        tensor_backward(s);
        optimizer_step(opt1);
        tensor_free(s);
    }
    tensor_to_doubles(W1, scalar_w);
    optimizer_free(opt1);
    param_clear();

    /* Phase 2: foreach path (default; env unset). */
    unsetenv("TAPE_OPTIMIZER_FOREACH");
    double wdata2[256];
    for (int i = 0; i < N; i++) wdata2[i] = (double)(i % 7) * 0.1 - 0.3;
    TensorHandle W2 = tensor_create_param_2d_f64(16, 16, heap_copy(wdata2, N));
    param_register("W", W2);
    OptimizerHandle opt2 = optimizer_create_adam(0.01, 0.9, 0.999, 1e-8);
    for (int step = 0; step < NSTEPS; step++) {
        optimizer_zero_grad(opt2);
        TensorHandle s = tensor_sum(W2);
        tensor_backward(s);
        optimizer_step(opt2);
        tensor_free(s);
    }
    tensor_to_doubles(W2, foreach_w);
    optimizer_free(opt2);
    param_clear();

    /* Compare element-wise. */
    double max_diff = 0.0;
    int max_idx = -1;
    for (int i = 0; i < N; i++) {
        double d = fabs(scalar_w[i] - foreach_w[i]);
        if (d > max_diff) { max_diff = d; max_idx = i; }
    }
    cr_assert(max_diff < 1e-12,
        "Adam foreach diverged from scalar: max_diff=%.6e at idx=%d (scalar=%.12g foreach=%.12g)",
        max_diff, max_idx, scalar_w[max_idx], foreach_w[max_idx]);
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

/* Non-learnable buffer (PyTorch register_buffer): a buffer must be SAVED +
   RELOADED by name like a param, but the optimizer must NEVER step it. To
   prove the is_buffer skip is load-bearing (and not just the no-grad guard),
   the buffer here is deliberately given requires_grad=1 and put in the loss,
   so it receives a real gradient — yet the step must leave it untouched. */
Test(training_optimizer_buffer, buffer_saved_but_not_stepped) {
    param_clear();
    /* idx 0 = learnable param w=1.0; idx 1 = buffer buf=5.0 (with a grad). */
    TensorHandle w = tensor_create_scalar(1.0, 1);
    param_register("buftest.w", w);
    TensorHandle buf = tensor_create_scalar(5.0, 1);
    param_register_buffer("buftest.buf", buf);

    cr_assert_eq(param_is_buffer(0), 0, "w is not a buffer");
    cr_assert_eq(param_is_buffer(1), 1, "buf is a buffer");

    /* loss = w*w + buf → grad_w = 2.0, grad_buf = 1.0. SGD(0.1): w -> 0.8;
       buf would -> 4.9 if stepped, must stay 5.0. */
    TensorHandle loss = tensor_add(tensor_mul(w, w), buf);
    OptimizerHandle opt = optimizer_create_sgd(0.1);
    native_train_step(opt, 0, 0.0, loss, 1.0);

    /* tol 1e-5: mlx scalars are F32, so 0.8 round-trips as 0.80000001. */
    cr_assert_float_eq(tensor_item(w), 0.8, 1e-5, "param stepped (got %.15f)", tensor_item(w));
    cr_assert_float_eq(tensor_item(buf), 5.0, 1e-5, "buffer NOT stepped (got %.15f)",
                       tensor_item(buf));
    optimizer_free(opt);

    /* Save (must include the buffer), wipe, re-register fresh, reload. */
    const char* path = "/tmp/idris-ml-c-buffer-roundtrip.safetensors";
    cr_assert_eq(param_save(path), 0, "param_save ok");
    param_clear();
    TensorHandle w2 = tensor_create_scalar(9.0, 1);
    param_register("buftest.w", w2);
    TensorHandle buf2 = tensor_create_scalar(9.0, 1);
    param_register_buffer("buftest.buf", buf2);
    cr_assert_eq(param_load(path), 0, "param_load ok");

    cr_assert_float_eq(tensor_item(buf2), 5.0, 1e-5, "buffer reloaded by name (got %.15f)",
                       tensor_item(buf2));
    cr_assert_float_eq(tensor_item(w2), 0.8, 1e-5, "param reloaded by name (got %.15f)",
                       tensor_item(w2));
    cr_assert_eq(param_is_buffer(1), 1, "buffer flag survives re-register");
    param_clear();
}

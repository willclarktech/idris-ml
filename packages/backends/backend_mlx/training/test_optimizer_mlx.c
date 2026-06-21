/* Criterion suite — mlx-specific optimizer paths.
 *
 * The common tape optimizer suite (test_optimizer_tape.c) already drives the
 * SGD / RMSprop / Adam / clip math through the public renamed FFI on every
 * backend. This file targets the mlx-ONLY uncovered branches in
 * backend_mlx/training/optimizer.cpp the common suite can't reach:
 *
 *   - the MLX_OPT_COMPILE=1 Adam compile path: get_adam_compiled +
 *     adam_step_compile + the compile branch in optimizer_step (lines
 *     168-310). Reachable only on mlx, only when MLX_OPT_COMPILE=1. Criterion
 *     forks each Test into a fresh process, so the setenv() takes effect
 *     before mlx_opt_compile_enabled() caches on first optimizer_step.
 *   - the AdamW (type 3) and Adam (type 2) eager step bodies, asserting the
 *     torch-matching update on mlx F32 storage.
 *   - optimizer_set_v's buffer-realloc branch (556-561): set_v before any
 *     step has allocated buffers.
 *   - optimizer_step_with_clip (643-653) + native_train_step_scaled clip
 *     dispatch (636, 638).
 *
 * Whole file is BACKEND_MLX-gated: these assert mlx F32 behavior and the
 * compile path is an mlx-local feature; tape/torch must not compile them.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* ----------------------------------------------------------------------
   Adam compile path (MLX_OPT_COMPILE=1). Drives get_adam_compiled +
   adam_step_compile + the compile branch in optimizer_step. One Adam step
   on loss = a*a (grad = 2a). The compiled and eager Adam paths share the
   torch.optim.Adam math, so the first-step update is:
     m1 = (1-b1)*g, v1 = (1-b2)*g^2
     mhat = m1/(1-b1), vhat = v1/(1-b2)  -> mhat = g, vhat = g^2
     step = lr * mhat / (sqrt(vhat) + eps) ~= lr * g / |g| = lr  (sign of g)
   so a -> a - lr (for g > 0).
   ---------------------------------------------------------------------- */
/* DISABLED: mlx's CPU-JIT (mx::compile) shells out to `g++`, which is absent in
   the nix dev shell (and the CI mlx runner) — clang only. The MLX_OPT_COMPILE
   Adam path can't be exercised here. Re-enable if a g++ (or clang-as-g++) lands
   on the mlx lane. */
Test(mlx_optimizer_compile, adam_compiled_step_matches_eager, .disabled = true) {
	setenv("MLX_OPT_COMPILE", "1", 1);
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("cmp_a", a);
	double lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8;
	OptimizerHandle opt = optimizer_create_adam(lr, b1, b2, eps);
	TensorHandle loss = tensor_mul(a, a); /* grad = 2*2 = 4 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	/* First Adam step magnitude ~= lr (bias-corrected, g > 0). */
	double g = 4.0;
	double m1 = (1.0 - b1) * g;
	double v1 = (1.0 - b2) * g * g;
	double mhat = m1 / (1.0 - b1);
	double vhat = v1 / (1.0 - b2);
	double step = lr * mhat / (sqrt(vhat) + eps);
	double expect = 2.0 - step;
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "compiled Adam step (got %.6f, want %.6f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
	unsetenv("MLX_OPT_COMPILE");
}

/* Compile path with two params, exercising the per-param scatter/gather loop
   in adam_step_compile (n == 2) and the get_adam_compiled cache key on n. */
/* DISABLED: mlx CPU-JIT needs g++ (absent in nix shell / CI mlx runner). */
Test(mlx_optimizer_compile, adam_compiled_two_params, .disabled = true) {
	setenv("MLX_OPT_COMPILE", "1", 1);
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	TensorHandle b = tensor_create_scalar(-3.0, 1);
	param_register("cmp2_a", a);
	param_register("cmp2_b", b);
	double lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8;
	OptimizerHandle opt = optimizer_create_adam(lr, b1, b2, eps);
	/* loss = a*a + b*b => grad(a)=2a=4, grad(b)=2b=-6 */
	TensorHandle loss = tensor_add(tensor_mul(a, a), tensor_mul(b, b));
	native_train_step(opt, 0, 0.0, loss, 0.0);
	/* First-step magnitude ~= lr toward the negative-gradient direction. */
	cr_assert_float_eq(tensor_item(a), 2.0 - lr, 2e-3, "compiled Adam a (got %.6f)",
	                   tensor_item(a));
	cr_assert_float_eq(tensor_item(b), -3.0 + lr, 2e-3, "compiled Adam b (got %.6f)",
	                   tensor_item(b));
	optimizer_free(opt);
	param_clear();
	unsetenv("MLX_OPT_COMPILE");
}

/* ----------------------------------------------------------------------
   AdamW eager step (type 3). Decoupled weight decay applied to the PRE-step
   weight, then the Adam update — torch.optim.AdamW order.
     w' = w - lr*wd*w   (decoupled decay)
     w'' = w' - lr * mhat / (sqrt(vhat) + eps)
   First step: mhat = g, vhat = g^2.
   ---------------------------------------------------------------------- */
Test(mlx_optimizer_eager, adamw_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("adamw_a", a);
	double lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
	OptimizerHandle opt = optimizer_create_adamw(lr, b1, b2, eps, wd);
	TensorHandle loss = tensor_mul(a, a); /* grad = 4 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	double g = 4.0;
	double mhat = g, vhat = g * g;
	double after_decay = 2.0 - lr * wd * 2.0;
	double expect = after_decay - lr * mhat / (sqrt(vhat) + eps);
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "AdamW decoupled-decay step (got %.6f, want %.6f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* Adam eager step (type 2, no compile) — the non-compile case 2 body. */
Test(mlx_optimizer_eager, adam_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("adam_a", a);
	double lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8;
	OptimizerHandle opt = optimizer_create_adam(lr, b1, b2, eps);
	TensorHandle loss = tensor_mul(a, a); /* grad = 4 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	double g = 4.0;
	double mhat = g, vhat = g * g;
	double expect = 2.0 - lr * mhat / (sqrt(vhat) + eps);
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "Adam eager step (got %.6f, want %.6f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* RMSprop WITH momentum — the momentum>0 branch of the eager step. */
Test(mlx_optimizer_eager, rmsprop_with_momentum) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("rmsm_a", a);
	double lr = 0.1, alpha = 0.99, eps = 1e-8, wd = 0.0, momentum = 0.9;
	OptimizerHandle opt = optimizer_create_rmsprop(lr, alpha, eps, wd, momentum);
	TensorHandle loss = tensor_mul(a, a); /* grad = 2 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	/* v = (1-alpha)*g^2 = 0.01*4 = 0.04; avg = sqrt(0.04)+eps = 0.2
	   buf = momentum*0 + g/avg = 2/0.2 = 10; step = lr*buf = 1.0 => a = 0.0 */
	double avg = sqrt(0.04) + eps;
	double buf = 2.0 / avg;
	double expect = 1.0 - lr * buf;
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "RMSprop momentum step (got %.6f, want %.6f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   optimizer_set_v before any step has allocated buffers (556-561): the
   size-mismatch realloc branch zeroes m/v then writes the supplied v.
   Read it back via optimizer_get_v. Mirror branch in optimizer_set_m.
   ---------------------------------------------------------------------- */
Test(mlx_optimizer_state, set_v_allocates_then_writes) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("setv_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double v_in[1] = {0.25};
	optimizer_set_v(opt, 0, v_in); /* buffers unallocated -> realloc branch */
	double v_out = -1.0;
	optimizer_get_v(opt, 0, &v_out);
	cr_assert_float_eq(v_out, 0.25, TEST_TOL_RELAXED, "set_v realloc+write (got %.6f)", v_out);
	/* m should have been zero-filled by the same realloc branch. */
	double m_out = -1.0;
	optimizer_get_m(opt, 0, &m_out);
	cr_assert_float_eq(m_out, 0.0, TEST_TOL_RELAXED, "m zero after set_v realloc (got %.6f)",
	                   m_out);
	optimizer_free(opt);
	param_clear();
}

Test(mlx_optimizer_state, set_m_allocates_then_writes) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("setm_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double m_in[1] = {0.5};
	optimizer_set_m(opt, 0, m_in); /* buffers unallocated -> realloc branch */
	double m_out = -1.0;
	optimizer_get_m(opt, 0, &m_out);
	cr_assert_float_eq(m_out, 0.5, TEST_TOL_RELAXED, "set_m realloc+write (got %.6f)", m_out);
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   optimizer_step_with_clip (643-653): clip-mode dispatch then step then
   zero_grad. clip_mode 1 (value) clamps grad before the SGD step. We must
   run backward ourselves first (this entry point does not call backward).
   grad(a) = 2*3 = 6 -> clamped to 2.0 -> SGD lr=0.1: a = 3 - 0.2 = 2.8.
   ---------------------------------------------------------------------- */
Test(mlx_optimizer_step_with_clip, value_clip_then_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(3.0, 1);
	param_register("swc_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 6 */
	tensor_backward(loss);
	optimizer_step_with_clip(opt, 1, 2.0, 0);
	cr_assert_float_eq(tensor_item(a), 2.8, TEST_TOL_RELAXED, "value-clip then SGD step (got %.6f)",
	                   tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

/* optimizer_step_with_clip, clip_mode 2 (norm). grad(a)=2*4=8, norm 8 >
   max_norm 2 -> scale 0.25 -> grad 2.0 -> SGD lr=0.1: a = 4 - 0.2 = 3.8. */
Test(mlx_optimizer_step_with_clip, norm_clip_then_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("swcn_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 8 */
	tensor_backward(loss);
	optimizer_step_with_clip(opt, 2, 2.0, 0);
	cr_assert_float_eq(tensor_item(a), 3.8, TEST_TOL_RELAXED, "norm-clip then SGD step (got %.6f)",
	                   tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   native_train_step_scaled with clip_mode 1 (636): unscale grads by
   1/scale, then value-clip, then step. scale = 2 -> grad(a)=2a=6 unscaled
   to 3.0, clamp to 2.0, SGD lr=0.1: a = 3 - 0.2 = 2.8. Returns
   loss_val/scale. All finite -> no NaN.
   ---------------------------------------------------------------------- */
Test(mlx_optimizer_scaled, scaled_value_clip_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(3.0, 1);
	param_register("scl_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 6 */
	double scale = 2.0;
	double ret = native_train_step_scaled(opt, 1, 2.0, loss, 8.0, scale);
	/* grad 6 unscaled /2 = 3.0, clamped to 2.0, SGD -> 3 - 0.2 = 2.8 */
	cr_assert_float_eq(tensor_item(a), 2.8, TEST_TOL_RELAXED, "scaled value-clip step (got %.6f)",
	                   tensor_item(a));
	cr_assert_float_eq(ret, 8.0 / scale, TEST_TOL_RELAXED, "returns loss/scale (got %.6f)", ret);
	optimizer_free(opt);
	param_clear();
}

/* native_train_step_scaled with clip_mode 2 (638): norm-clip after unscale.
   scale = 2 -> grad(a)=2*8=16 unscaled to 8.0, norm 8 > max 2 -> scale 0.25
   -> grad 2.0, SGD lr=0.1: a = 8 - 0.2 = 7.8. */
Test(mlx_optimizer_scaled, scaled_norm_clip_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(8.0, 1);
	param_register("scln_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 16 */
	double scale = 2.0;
	double ret = native_train_step_scaled(opt, 2, 2.0, loss, 4.0, scale);
	cr_assert_float_eq(tensor_item(a), 7.8, TEST_TOL_RELAXED, "scaled norm-clip step (got %.6f)",
	                   tensor_item(a));
	cr_assert_float_eq(ret, 4.0 / scale, TEST_TOL_RELAXED, "returns loss/scale (got %.6f)", ret);
	optimizer_free(opt);
	param_clear();
}

#endif /* BACKEND_MLX */

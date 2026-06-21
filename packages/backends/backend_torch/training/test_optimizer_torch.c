/* Criterion suite — torch-specific optimizer paths.
 *
 * The common tape optimizer suite (test_optimizer_tape.c) runs on every backend
 * and already drives, on torch: SGD (via the clip tests), RMSprop no-momentum,
 * clip-value, clip-norm-below-threshold, and the get_m/get_v pre-alloc zero-fill.
 *
 * This file targets the torch fused-foreach paths in
 * backend_torch/training/optimizer.cpp the common suite does NOT reach:
 *   - adam_step_foreach + adam_core_foreach (no common Adam *step* test).
 *   - adamw_step_foreach + optimizer_create_adamw (decoupled weight decay).
 *   - rmsprop_step_foreach momentum branch + weight-decay branch.
 *   - clip_grad_norm rescale branch (the common rescale test is BACKEND_TAPE-only).
 *   - optimizer_set_param_lr + the per-LR bucketing in optimizer_step.
 *   - optimizer_set_m/get_m + set_v/get_v create-state round-trip.
 *   - optimizer_get_meta / optimizer_set_meta round-trip.
 *   - optimizer_step_with_clip + native_train_step_scaled.
 *
 * Whole file is BACKEND_TORCH-gated: torch CPU storage is F64, so the oracles
 * assert at F64 tolerance; tape/mlx must not compile them (mlx is F32; tape's
 * loop math is asserted in the common suite).
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* ----------------------------------------------------------------------
   Adam eager foreach step (type 2). loss = a*a => grad = 2a = 4 at a = 2.
   First step (bias-corrected): mhat = g, vhat = g^2, so
     a -> a - lr * g / (sqrt(g^2) + eps).
   ---------------------------------------------------------------------- */
Test(torch_optimizer, adam_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("t_adam_a", a);
	double lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8;
	OptimizerHandle opt = optimizer_create_adam(lr, b1, b2, eps);
	TensorHandle loss = tensor_mul(a, a); /* grad = 4 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	double g = 4.0;
	double mhat = g, vhat = g * g;
	double expect = 2.0 - lr * mhat / (sqrt(vhat) + eps);
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "Adam foreach step (got %.9f, want %.9f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* AdamW eager foreach step (type 3): decoupled weight decay on the PRE-step
   weight, then the Adam update — torch.optim.AdamW order.
     w' = w - lr*wd*w   then   w'' = w' - lr * mhat / (sqrt(vhat) + eps). */
Test(torch_optimizer, adamw_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("t_adamw_a", a);
	double lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
	OptimizerHandle opt = optimizer_create_adamw(lr, b1, b2, eps, wd);
	TensorHandle loss = tensor_mul(a, a); /* grad = 4 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	double g = 4.0;
	double mhat = g, vhat = g * g;
	double after_decay = 2.0 - lr * wd * 2.0;
	double expect = after_decay - lr * mhat / (sqrt(vhat) + eps);
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "AdamW decoupled-decay step (got %.9f, want %.9f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* RMSprop WITH momentum — the use_momentum branch (345-348).
   v = (1-alpha)*g^2; avg = sqrt(v)+eps; buf = momentum*0 + g/avg; step = lr*buf. */
Test(torch_optimizer, rmsprop_with_momentum) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_rmsm_a", a);
	double lr = 0.1, alpha = 0.99, eps = 1e-8, wd = 0.0, momentum = 0.9;
	OptimizerHandle opt = optimizer_create_rmsprop(lr, alpha, eps, wd, momentum);
	TensorHandle loss = tensor_mul(a, a); /* grad = 2 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	double g = 2.0;
	double v = (1.0 - alpha) * g * g;
	double avg = sqrt(v) + eps;
	double buf = g / avg;
	double expect = 1.0 - lr * buf;
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "RMSprop momentum step (got %.9f, want %.9f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* RMSprop WITH weight decay, no momentum — the use_wd branch (330-334):
   g_eff = g + wd*p; v = (1-alpha)*g_eff^2; avg = sqrt(v)+eps;
   w -= lr * g_eff / avg. */
Test(torch_optimizer, rmsprop_weight_decay) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("t_rmswd_a", a);
	double lr = 0.1, alpha = 0.99, eps = 1e-8, wd = 0.05, momentum = 0.0;
	OptimizerHandle opt = optimizer_create_rmsprop(lr, alpha, eps, wd, momentum);
	TensorHandle loss = tensor_mul(a, a); /* grad = 4 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	double g_eff = 4.0 + wd * 2.0; /* g + wd*p */
	double v = (1.0 - alpha) * g_eff * g_eff;
	double avg = sqrt(v) + eps;
	double expect = 2.0 - lr * g_eff / avg;
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "RMSprop weight-decay step (got %.9f, want %.9f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* clip_grad_norm rescale branch (clip_mode 2, norm > max_norm). grad(a)=2*4=8.
   The common rescale test is BACKEND_TAPE-only because torch's
   torch::nn::utils::clip_grad_norm_ adds a 1e-6 eps to the denominator:
   clip_coef = max_norm / (total_norm + 1e-6), so the rescaled grad is just
   under 2.0 and the post-step value is just over 3.8. Encode the eps-aware
   oracle (tape's clip has no eps and asserts an exact 3.8). */
Test(torch_optimizer, clip_grad_norm_rescales) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("t_clipn_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a);      /* grad = 8 */
	native_train_step(opt, 2, 2.0, loss, 0.0); /* clip_mode 2, max_norm 2.0 */
	double clip_coef = 2.0 / (8.0 + 1e-6);
	double expect = 4.0 - 0.1 * (8.0 * clip_coef);
	cr_assert_float_eq(tensor_item(a), expect, 1e-6,
	                   "grad-norm 8 rescaled then SGD (got %.9f, want %.9f)", tensor_item(a),
	                   expect);
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   optimizer_set_param_lr + the per-LR bucketing in optimizer_step (417-435).
   Two params a,b (loss = a*a + b*b => grad 2 each). Base SGD lr 0.1; override
   b's LR to 0.5. After one step: a = 1 - 0.1*2 = 0.8; b = 1 - 0.5*2 = 0.0.
   ---------------------------------------------------------------------- */
Test(torch_optimizer, set_param_lr_buckets) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	TensorHandle b = tensor_create_scalar(1.0, 1);
	param_register("t_plr_a", a);
	param_register("t_plr_b", b);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	optimizer_set_param_lr(opt, "t_plr_b", 0.5);
	TensorHandle loss = tensor_add(tensor_mul(a, a), tensor_mul(b, b)); /* grad 2 each */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	cr_assert_float_eq(tensor_item(a), 0.8, TEST_TOL_RELAXED, "base-LR param a (got %.9f)",
	                   tensor_item(a));
	cr_assert_float_eq(tensor_item(b), 0.0, TEST_TOL_RELAXED, "override-LR param b (got %.9f)",
	                   tensor_item(b));
	optimizer_free(opt);
	param_clear();
}

/* set_param_lr with lr < 0 clears an override, restoring the base LR. */
Test(torch_optimizer, set_param_lr_clear_restores_base) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_plrc_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	optimizer_set_param_lr(opt, "t_plrc_a", 0.5);
	optimizer_set_param_lr(opt, "t_plrc_a", -1.0); /* clear -> base 0.1 */
	TensorHandle loss = tensor_mul(a, a);          /* grad 2 */
	native_train_step(opt, 0, 0.0, loss, 0.0);
	cr_assert_float_eq(tensor_item(a), 0.8, TEST_TOL_RELAXED,
	                   "cleared override uses base LR -> 0.8 (got %.9f)", tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   optimizer_set_m/get_m + set_v/get_v create-state round-trip (Adam, type 2).
   No step has run; set_m/set_v create the AdamParamState then write the buffer.
   ---------------------------------------------------------------------- */
Test(torch_optimizer_state, set_get_m_v_roundtrip) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_state_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double m_in[1] = {0.5};
	double v_in[1] = {0.25};
	optimizer_set_m(opt, 0, m_in); /* creates state, writes exp_avg */
	optimizer_set_v(opt, 0, v_in); /* writes exp_avg_sq */
	double m_out = -1.0, v_out = -1.0;
	optimizer_get_m(opt, 0, &m_out);
	optimizer_get_v(opt, 0, &v_out);
	cr_assert_float_eq(m_out, 0.5, TEST_TOL_TIGHT, "set/get m round-trip (got %.9f)", m_out);
	cr_assert_float_eq(v_out, 0.25, TEST_TOL_TIGHT, "set/get v round-trip (got %.9f)", v_out);
	optimizer_free(opt);
	param_clear();
}

/* RMSprop (type 1) state round-trip — exercises the type==1 arms of set_m
   (momentum_buffer), set_v (square_avg), and get_m/get_v that the Adam
   round-trip above does not reach. */
Test(torch_optimizer_state, rmsprop_set_get_m_v_roundtrip) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_rstate_a", a);
	OptimizerHandle opt = optimizer_create_rmsprop(0.01, 0.99, 1e-8, 0.0, 0.9);
	double m_in[1] = {0.3}; /* momentum_buffer */
	double v_in[1] = {0.7}; /* square_avg */
	optimizer_set_m(opt, 0, m_in);
	optimizer_set_v(opt, 0, v_in);
	double m_out = -1.0, v_out = -1.0;
	optimizer_get_m(opt, 0, &m_out);
	optimizer_get_v(opt, 0, &v_out);
	cr_assert_float_eq(m_out, 0.3, TEST_TOL_TIGHT, "rmsprop set/get m (got %.9f)", m_out);
	cr_assert_float_eq(v_out, 0.7, TEST_TOL_TIGHT, "rmsprop set/get v (got %.9f)", v_out);
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   optimizer_get_meta / optimizer_set_meta round-trip of the 8 hyperparameters
   (slot 8 is step, which is 0 with no param state).
   ---------------------------------------------------------------------- */
Test(torch_optimizer_state, meta_roundtrip) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_meta_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	/* meta = [type, lr, b1, b2, eps, alpha, wd, momentum, step] */
	double in9[9] = {2.0, 0.123, 0.8, 0.95, 1e-7, 0.97, 0.02, 0.0, 0.0};
	optimizer_set_meta(opt, in9);
	double out9[9] = {0};
	optimizer_get_meta(opt, out9);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(out9[i], in9[i], TEST_TOL_TIGHT, "meta slot %d round-trip (got %.9f)", i,
		                   out9[i]);
	}
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   optimizer_step_with_clip (does NOT call backward — we run it first).
   clip_mode 1 (value): grad(a)=2*3=6 -> clamp to 2.0 -> SGD lr=0.1: 3 - 0.2 = 2.8.
   ---------------------------------------------------------------------- */
Test(torch_optimizer, step_with_clip_value) {
	param_clear();
	TensorHandle a = tensor_create_scalar(3.0, 1);
	param_register("t_swc_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 6 */
	tensor_backward(loss);
	optimizer_step_with_clip(opt, 1, 2.0, 0);
	cr_assert_float_eq(tensor_item(a), 2.8, TEST_TOL_RELAXED, "value-clip then SGD (got %.9f)",
	                   tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

/* native_train_step_scaled, clip_mode 1: unscale grads by 1/scale then value-clip
   then step. scale=2 -> grad 6 unscaled to 3.0, clamp to 2.0, SGD -> 3 - 0.2 = 2.8.
   Returns loss_val/scale. */
Test(torch_optimizer, scaled_value_clip_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(3.0, 1);
	param_register("t_scl_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 6 */
	double scale = 2.0;
	double ret = native_train_step_scaled(opt, 1, 2.0, loss, 8.0, scale);
	cr_assert_float_eq(tensor_item(a), 2.8, TEST_TOL_RELAXED, "scaled value-clip step (got %.9f)",
	                   tensor_item(a));
	cr_assert_float_eq(ret, 8.0 / scale, TEST_TOL_RELAXED, "returns loss/scale (got %.9f)", ret);
	optimizer_free(opt);
	param_clear();
}

#endif /* BACKEND_TORCH */

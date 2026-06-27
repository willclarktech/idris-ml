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

/* ----------------------------------------------------------------------
   optimizer_set_lr per type — each switch arm (SGD/RMSprop/Adam/AdamW) in
   optimizer_set_lr updates that optimizer's option object. get_meta[1]
   mirrors w->lr, set in the same call.
   ---------------------------------------------------------------------- */
Test(torch_optimizer, set_lr_per_type) {
	double m[9];
	param_clear();
	TensorHandle a0 = tensor_create_scalar(1.0, 1);
	param_register("t_slr0", a0);
	OptimizerHandle o0 = optimizer_create_sgd(0.1);
	optimizer_set_lr(o0, 0.5);
	optimizer_get_meta(o0, m);
	cr_assert_float_eq(m[1], 0.5, TEST_TOL_TIGHT, "SGD set_lr (got %.9f)", m[1]);
	optimizer_free(o0);
	param_clear();

	TensorHandle a1 = tensor_create_scalar(1.0, 1);
	param_register("t_slr1", a1);
	OptimizerHandle o1 = optimizer_create_rmsprop(0.1, 0.99, 1e-8, 0.0, 0.0);
	optimizer_set_lr(o1, 0.6);
	optimizer_get_meta(o1, m);
	cr_assert_float_eq(m[1], 0.6, TEST_TOL_TIGHT, "RMSprop set_lr (got %.9f)", m[1]);
	optimizer_free(o1);
	param_clear();

	TensorHandle a2 = tensor_create_scalar(1.0, 1);
	param_register("t_slr2", a2);
	OptimizerHandle o2 = optimizer_create_adam(0.1, 0.9, 0.999, 1e-8);
	optimizer_set_lr(o2, 0.7);
	optimizer_get_meta(o2, m);
	cr_assert_float_eq(m[1], 0.7, TEST_TOL_TIGHT, "Adam set_lr (got %.9f)", m[1]);
	optimizer_free(o2);
	param_clear();

	TensorHandle a3 = tensor_create_scalar(1.0, 1);
	param_register("t_slr3", a3);
	OptimizerHandle o3 = optimizer_create_adamw(0.1, 0.9, 0.999, 1e-8, 0.01);
	optimizer_set_lr(o3, 0.8);
	optimizer_get_meta(o3, m);
	cr_assert_float_eq(m[1], 0.8, TEST_TOL_TIGHT, "AdamW set_lr (got %.9f)", m[1]);
	optimizer_free(o3);
	param_clear();
}

/* get_m / get_v on AdamW (type 3): after a step the param has an AdamParamState,
   but get_m/get_v only special-case type==2 (Adam) and type==1 (RMSprop), so
   type 3 falls to the "no moment buffer -> zeros" else arm. */
Test(torch_optimizer_state, get_m_v_adamw_else_zeros) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 1);
	param_register("t_gmw_a", a);
	OptimizerHandle opt = optimizer_create_adamw(0.1, 0.9, 0.999, 1e-8, 0.01);
	TensorHandle loss = tensor_mul(a, a);
	native_train_step(opt, 0, 0.0, loss, 0.0); /* creates per-param state */
	double m_out = -1.0, v_out = -1.0;
	optimizer_get_m(opt, 0, &m_out);
	optimizer_get_v(opt, 0, &v_out);
	cr_assert_float_eq(m_out, 0.0, TEST_TOL_TIGHT, "AdamW get_m else -> 0 (got %.9f)", m_out);
	cr_assert_float_eq(v_out, 0.0, TEST_TOL_TIGHT, "AdamW get_v else -> 0 (got %.9f)", v_out);
	optimizer_free(opt);
	param_clear();
}

/* set_m / set_v on SGD (type 0): no Adam/RMSprop state to create, so both take
   the `else return;` no-op arm. A subsequent get_m reads back zeros. */
Test(torch_optimizer_state, set_m_v_sgd_noop) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_sms_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	double buf[1] = {0.5};
	optimizer_set_m(opt, 0, buf); /* type 0 -> else return (no-op) */
	optimizer_set_v(opt, 0, buf); /* type 0 -> else return (no-op) */
	double m_out = -1.0;
	optimizer_get_m(opt, 0, &m_out);
	cr_assert_float_eq(m_out, 0.0, TEST_TOL_TIGHT, "SGD set_m is a no-op (got %.9f)", m_out);
	optimizer_free(opt);
	param_clear();
}

/* set_v on a FRESH optimizer (no prior set_m) creates the param state via
   set_v's own create branch — the round-trip tests above always call set_m
   first, so set_v sees existing state and skips its create arm. */
Test(torch_optimizer_state, set_v_first_creates_state) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_svf_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double v_in[1] = {0.4};
	optimizer_set_v(opt, 0, v_in); /* creates AdamParamState, writes exp_avg_sq */
	double v_out = -1.0;
	optimizer_get_v(opt, 0, &v_out);
	cr_assert_float_eq(v_out, 0.4, TEST_TOL_TIGHT, "set_v-first round-trip (got %.9f)", v_out);
	optimizer_free(opt);
	param_clear();
}

/* RMSprop with momentum=0: get_m sees state (created by set_v) but no
   momentum_buffer, so it takes the zeros_like fallback. */
Test(torch_optimizer_state, get_m_rmsprop_no_momentum_zeros) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_gmr_a", a);
	OptimizerHandle opt = optimizer_create_rmsprop(0.01, 0.99, 1e-8, 0.0, /*momentum=*/0.0);
	double v_in[1] = {0.7};
	optimizer_set_v(opt, 0, v_in); /* creates RMSpropParamState (no momentum_buffer) */
	double m_out = -1.0;
	optimizer_get_m(opt, 0, &m_out); /* momentum_buffer undefined -> zeros_like fallback */
	cr_assert_float_eq(m_out, 0.0, TEST_TOL_TIGHT, "RMSprop no-momentum get_m -> 0 (got %.9f)",
	                   m_out);
	optimizer_free(opt);
	param_clear();
}

/* get_meta/set_meta step round-trip WITH live param state (set_m creates it):
   exercises the step read (get_meta) + step update (set_meta) on an existing
   AdamParamState — the meta_roundtrip test above has no state so skips those. */
Test(torch_optimizer_state, meta_step_with_state_adam) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_msa_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double mbuf[1] = {0.1};
	optimizer_set_m(opt, 0, mbuf); /* creates AdamParamState (step = pending = 0) */
	double m[9];
	optimizer_get_meta(opt, m);
	cr_assert_float_eq(m[8], 0.0, TEST_TOL_TIGHT, "Adam step read pre-set (got %.9f)", m[8]);
	double in9[9] = {2.0, 0.001, 0.9, 0.999, 1e-8, 0.0, 0.0, 0.0, 5.0}; /* step=5 */
	optimizer_set_meta(opt, in9); /* updates existing state step */
	optimizer_get_meta(opt, m);
	cr_assert_float_eq(m[8], 5.0, TEST_TOL_TIGHT, "Adam step after set_meta (got %.9f)", m[8]);
	optimizer_free(opt);
	param_clear();
}

/* Same, RMSprop (type 1): covers the RMSpropParamState step read/update arms. */
Test(torch_optimizer_state, meta_step_with_state_rmsprop) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("t_msr_a", a);
	OptimizerHandle opt = optimizer_create_rmsprop(0.01, 0.99, 1e-8, 0.0, 0.9);
	double mbuf[1] = {0.2};
	optimizer_set_m(opt, 0, mbuf); /* creates RMSpropParamState */
	double in9[9] = {1.0, 0.01, 0.0, 0.0, 1e-8, 0.99, 0.0, 0.9, 7.0}; /* type=1, step=7 */
	optimizer_set_meta(opt, in9);
	double m[9];
	optimizer_get_meta(opt, m);
	cr_assert_float_eq(m[8], 7.0, TEST_TOL_TIGHT, "RMSprop step after set_meta (got %.9f)", m[8]);
	optimizer_free(opt);
	param_clear();
}

/* optimizer_step_with_clip, clip_mode 2 (norm). grad(a)=8; torch clip_grad_norm_
   adds 1e-6 eps to the denominator (same eps-aware oracle as the native_train_step
   norm test above). */
Test(torch_optimizer, step_with_clip_norm) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("t_swcn_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 8 */
	tensor_backward(loss);
	optimizer_step_with_clip(opt, 2, 2.0, 0); /* norm clip, max_norm 2.0 */
	double clip_coef = 2.0 / (8.0 + 1e-6);
	double expect = 4.0 - 0.1 * (8.0 * clip_coef);
	cr_assert_float_eq(tensor_item(a), expect, 1e-6, "norm-clip then SGD (got %.9f, want %.9f)",
	                   tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

/* native_train_step_scaled, clip_mode 2 (norm). scale=2 unscales grad 8 -> 4,
   then norm-clip to max 2.0 (eps-aware), then SGD. */
Test(torch_optimizer, scaled_norm_clip_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("t_scn_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 8 (scaled) */
	double scale = 2.0;
	native_train_step_scaled(opt, 2, 2.0, loss, 8.0, scale); /* unscale -> grad 4, norm-clip */
	double clip_coef = 2.0 / (4.0 + 1e-6);
	double expect = 4.0 - 0.1 * (4.0 * clip_coef);
	cr_assert_float_eq(tensor_item(a), expect, 1e-5, "scaled norm-clip step (got %.9f, want %.9f)",
	                   tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

#endif /* BACKEND_TORCH */

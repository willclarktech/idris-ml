/* shared/training/optimizer.c — backend-agnostic optimizer surface.
 *
 * The FFI-named optimizer entry points the Idris training loop binds
 * against. Owns:
 *   - tiny trampolines that hand the backend-specific construction /
 *     step / serialization off to `g_active_port.optimizer_*` (each
 *     backend supplies a struct laid out for its own native math —
 *     tape's flat-buffer Optimizer, torch's libtorch OptWrapper,
 *     mlx's TBD);
 *   - cross-cutting helpers that don't touch optimizer state and so
 *     stay genuinely shared: `optimizer_zero_grad` (delegates to
 *     `param_zero_all_grads`), `polyak_blend_pair`, `optimizer_clip_*`,
 *     and the `native_train_step` / `optimizer_step_with_clip`
 *     high-level wrappers.
 *
 * Compiled once per backend in TRAINING_ADAPTER_BACKENDS with that
 * backend's rename header so the unsuffixed names (`optimizer_create_adam`
 * etc.) resolve to backend-suffixed exports the Idris-side bindings
 * already target.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "port.h"
#include "../../backend.h"

/* ----------------------------------------------------------------------
   Constructors + lifecycle — port trampolines.
   ---------------------------------------------------------------------- */

OptimizerHandle optimizer_create_sgd(double lr) {
	return (OptimizerHandle)g_active_port.optimizer_create_sgd(lr);
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps, double weight_decay,
                                         double momentum) {
	return (OptimizerHandle)g_active_port.optimizer_create_rmsprop(lr, alpha, eps, weight_decay,
	                                                               momentum);
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
	return (OptimizerHandle)g_active_port.optimizer_create_adam(lr, beta1, beta2, eps);
}

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
	return (OptimizerHandle)g_active_port.optimizer_create_adamw(lr, beta1, beta2, eps,
	                                                             weight_decay);
}

void optimizer_free(OptimizerHandle h) {
	g_active_port.optimizer_free((void*)h);
}

void optimizer_set_lr(OptimizerHandle h, double lr) {
	g_active_port.optimizer_set_lr((void*)h, lr);
}

void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
	g_active_port.optimizer_set_param_lr((void*)h, name, lr);
}

void optimizer_own_param(OptimizerHandle h, const char* name) {
	g_active_port.optimizer_own_param((void*)h, name);
}

/* ----------------------------------------------------------------------
   Step — port trampoline. Adapter is responsible for ALL backend
   hygiene (intermediate cleanup, prof_* updates, tape_reset etc.).
   ---------------------------------------------------------------------- */
void optimizer_step(OptimizerHandle h) {
	g_active_port.optimizer_step((void*)h);
}

/* ----------------------------------------------------------------------
   Genuinely shared helpers — touch no optimizer state, only the param
   registry + the port's per-element accessors. Identical math across
   backends.
   ---------------------------------------------------------------------- */

void optimizer_zero_grad(OptimizerHandle h) {
	(void)h;
	param_zero_all_grads();
}

int polyak_blend_pair(double tau, const char* online_name, const char* target_name) {
	if (!online_name || !target_name) return 0;
	void* on_t = NULL;
	void* tg_t = NULL;
	/* Exact-match both names in a single registry pass — no prefix logic,
	   so a name that is a proper prefix of another can't over-match. */
	for (int i = 0; i < param_count(); i++) {
		const char* nm = param_name(i);
		if (!on_t && strcmp(nm, online_name) == 0) on_t = param_tensor(i);
		if (!tg_t && strcmp(nm, target_name) == 0) tg_t = param_tensor(i);
	}
	if (!on_t || !tg_t) return 0;
	int n_on = g_active_port.tensor_numel(on_t);
	if (n_on != g_active_port.tensor_numel(tg_t)) return 0;
	double one_minus_tau = 1.0 - tau;
	for (int k = 0; k < n_on; k++) {
		double tg = g_active_port.data_read(tg_t, k);
		double on = g_active_port.data_read(on_t, k);
		g_active_port.data_write(tg_t, k, one_minus_tau * tg + tau * on);
	}
	return 1;
}

void optimizer_clip_grad_value(double max_val) {
	for (int i = 0; i < param_count(); i++) {
		if (param_is_buffer(i)) continue; /* buffers carry no grad — never clipped */
		void* t = param_tensor(i);
		if (!g_active_port.tensor_has_grad(t)) continue;
		int n = g_active_port.tensor_numel(t);
		for (int j = 0; j < n; j++) {
			double v = g_active_port.grad_read(t, j);
			if (v > max_val)
				g_active_port.grad_write(t, j, max_val);
			else if (v < -max_val)
				g_active_port.grad_write(t, j, -max_val);
		}
	}
}

double optimizer_clip_grad_norm(double max_norm) {
	double total = 0;
	for (int i = 0; i < param_count(); i++) {
		if (param_is_buffer(i)) continue; /* buffers contribute no grad norm */
		void* t = param_tensor(i);
		if (!g_active_port.tensor_has_grad(t)) continue;
		int n = g_active_port.tensor_numel(t);
		for (int j = 0; j < n; j++) {
			double v = g_active_port.grad_read(t, j);
			total += v * v;
		}
	}
	double norm = sqrt(total);
	if (norm > max_norm) {
		double scale = max_norm / norm;
		for (int i = 0; i < param_count(); i++) {
			if (param_is_buffer(i)) continue;
			void* t = param_tensor(i);
			if (!g_active_port.tensor_has_grad(t)) continue;
			int n = g_active_port.tensor_numel(t);
			for (int j = 0; j < n; j++) {
				double v = g_active_port.grad_read(t, j);
				g_active_port.grad_write(t, j, v * scale);
			}
		}
	}
	return norm;
}

/* ----------------------------------------------------------------------
   Serialization — port trampolines (each backend's serializer touches
   its own state representation).
   ---------------------------------------------------------------------- */

int optimizer_buf_count(OptimizerHandle h) {
	return g_active_port.optimizer_buf_count((void*)h);
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
	g_active_port.optimizer_get_m((void*)h, idx, out);
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
	g_active_port.optimizer_get_v((void*)h, idx, out);
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
	g_active_port.optimizer_set_m((void*)h, idx, data);
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
	g_active_port.optimizer_set_v((void*)h, idx, data);
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
	g_active_port.optimizer_get_meta((void*)h, out9);
}

void optimizer_set_meta(OptimizerHandle h, const double* in9) {
	g_active_port.optimizer_set_meta((void*)h, in9);
}

/* ----------------------------------------------------------------------
   High-level train-step wrappers — zero_grad → backward → clip → step.
   ---------------------------------------------------------------------- */

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val, TensorHandle loss_ptr,
                         double loss_val) {
	optimizer_zero_grad(opt);
	if (g_active_port.tensor_requires_grad((void*)loss_ptr))
		g_active_port.backward((void*)loss_ptr);
	/* Use the prefix-scoped clip variants so multi-optimizer training
	   (SAC's actor / q1 / q2) clips only this optimizer's owned params.
	   Single-optimizer cases (empty prefix) walk every registered
	   param — same effective result as the global clip. */
	if (clip_mode == 1)
		g_active_port.optimizer_clip_grad_value_filtered((void*)opt, clip_val);
	else if (clip_mode == 2)
		g_active_port.optimizer_clip_grad_norm_filtered((void*)opt, clip_val);
	optimizer_step(opt);
	return loss_val;
}

/* GradScaler-aware train step (A3 of the type-safe mixed-precision
   plan #410). The caller has already multiplied the loss by `scale`
   before backward — this function:
     1. zero_grad
     2. backward(scaled_loss) — grads at the scaled magnitude
     3. walk the registry: divide each grad by `scale`, checking for
        non-finite values (overflow from low-precision compute)
     4. if any non-finite: skip clip+step, return NaN — caller halves
        the scale and discards the update
     5. else: clip + step, return the *unscaled* loss
   The NaN-sentinel return lets the caller advance its own scale state
   machine (Idris-side IORef-based or equivalent) without an extra FFI
   roundtrip. */
double native_train_step_scaled(OptimizerHandle opt, int clip_mode, double clip_val,
                                TensorHandle loss_ptr, double loss_val, double scale) {
	optimizer_zero_grad(opt);
	if (g_active_port.tensor_requires_grad((void*)loss_ptr))
		g_active_port.backward((void*)loss_ptr);

	double inv_scale = 1.0 / scale;
	int has_nonfinite = 0;
	for (int i = 0; i < param_count(); i++) {
		if (param_is_buffer(i)) continue; /* buffers carry no grad to unscale */
		void* t = param_tensor(i);
		if (!g_active_port.tensor_has_grad(t)) continue;
		int n = g_active_port.tensor_numel(t);
		for (int j = 0; j < n; j++) {
			double v = g_active_port.grad_read(t, j);
			if (!isfinite(v)) has_nonfinite = 1;
			g_active_port.grad_write(t, j, v * inv_scale);
		}
	}

	if (has_nonfinite) {
		/* Signal the caller to skip + halve. NaN sentinel; callers
		   compare via isnan() (or Idris-side equivalent). The grads
		   have already been unscaled (mostly into +/-Inf or NaN), so
		   the optimizer would skip them or produce garbage — we
		   explicitly DO NOT call optimizer_step. */
		return (double)NAN;
	}

	if (clip_mode == 1)
		g_active_port.optimizer_clip_grad_value_filtered((void*)opt, clip_val);
	else if (clip_mode == 2)
		g_active_port.optimizer_clip_grad_norm_filtered((void*)opt, clip_val);
	optimizer_step(opt);
	return loss_val * inv_scale;
}

int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
	(void)dummy;
	if (clip_mode == 1)
		g_active_port.optimizer_clip_grad_value_filtered((void*)opt, clip_val);
	else if (clip_mode == 2)
		g_active_port.optimizer_clip_grad_norm_filtered((void*)opt, clip_val);
	optimizer_step(opt);
	optimizer_zero_grad(opt);
	return 0;
}

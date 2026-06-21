/* Native optimizer surface for the mlx backend.
 *
 * Owns the Optimizer struct (mlx version: m/v buffers held as mx::array,
 * per-param LR overrides, prefix scope for multi-optimizer SAC, step
 * counter), the per-op-loop step path (SGD/RMSprop/Adam/AdamW), the
 * optional mx::compile-cached Adam variant gated by MLX_OPT_COMPILE=1,
 * the clip-filtered helpers, polyak_blend, the optimizer-state get/set
 * accessors used by the safetensors serializer, and the two clip-scoped
 * train-step entry points (native_train_step, optimizer_step_with_clip).
 *
 * Math intentionally mirrors PyTorch's torch.optim semantics op-for-op:
 *   - RMSprop keeps lr OUTSIDE the momentum buffer so an LR schedule
 *     doesn't carry stale rates in the buffer.
 *   - AdamW applies decoupled weight-decay to the PRE-step weight, then
 *     the Adam update (param.mul_(1 - lr*wd) before addcdiv_), matching
 *     torch.optim.AdamW.
 *   - Optimizer-state scalars (beta1/beta2/eps/bc1/bc2 etc.) are hoisted
 *     out of the per-param loop so they don't graph-build per param. */
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../../backend.h"
#include "../tensor.h"
#include "../tape.h"      /* tape_reset, OP_CONST, tape_append */
#include "../precision.h" /* scalar_like, mx_to_doubles, mx_array_from_doubles */
#include "profiling.h"

extern "C" int param_count(void);
extern "C" void* param_tensor(int i);
extern "C" const char* param_name(int i);
extern "C" void param_zero_all_grads(void);
extern "C" void tensor_backward(TensorHandle h);
extern "C" int tensor_requires_grad(TensorHandle h);
extern "C" void _dbg_dump_param_grads_if_enabled_mlx(void);

struct Optimizer {
	int type = 0; // 0=sgd, 1=rmsprop, 2=adam, 3=adamw
	double lr = 0, beta1 = 0, beta2 = 0, eps = 0;
	double alpha = 0, weight_decay = 0, momentum = 0;
	int t = 0;
	/* Per-parameter buffers — held as mx::array so the optimizer math
	   stays inside mlx's autograd graph. */
	std::vector<mx::array> m_bufs, v_bufs;
	/* Per-param LR overrides (indexed by param registry position, -1 = use base lr). */
	std::vector<double> param_lr;
	std::unordered_set<std::string> owned; // empty = manages all params; else exact owned names
};

/* Empty owned-set => manages every param. Else owned iff the exact name is in
 * the set — no prefix logic, so `q1_` can't leak into `q1tgt_`. */
static bool opt_owns_param_mlx(Optimizer* opt, int i) {
	if (opt->owned.empty()) return true;
	return opt->owned.contains(param_name(i));
}

extern "C" OptimizerHandle optimizer_create_sgd(double lr) {
	auto* opt = new Optimizer();
	opt->type = 0;
	opt->lr = lr;
	opt->t = 0;
	return (OptimizerHandle)opt;
}

extern "C" OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                                    double weight_decay, double momentum) {
	auto* opt = new Optimizer();
	opt->type = 1;
	opt->lr = lr;
	opt->alpha = alpha;
	opt->eps = eps;
	opt->weight_decay = weight_decay;
	opt->momentum = momentum;
	opt->t = 0;
	return (OptimizerHandle)opt;
}

extern "C" OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2,
                                                 double eps) {
	auto* opt = new Optimizer();
	opt->type = 2;
	opt->lr = lr;
	opt->beta1 = beta1;
	opt->beta2 = beta2;
	opt->eps = eps;
	opt->t = 0;
	return (OptimizerHandle)opt;
}

extern "C" OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                                  double weight_decay) {
	auto* opt = new Optimizer();
	opt->type = 3;
	opt->lr = lr;
	opt->beta1 = beta1;
	opt->beta2 = beta2;
	opt->eps = eps;
	opt->weight_decay = weight_decay;
	opt->t = 0;
	return (OptimizerHandle)opt;
}

extern "C" void optimizer_free(OptimizerHandle h) {
	delete (Optimizer*)h;
}
extern "C" void optimizer_zero_grad(OptimizerHandle h) {
	(void)h;
	param_zero_all_grads();
}

extern "C" void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
	auto* opt = (Optimizer*)h;
	int const np = param_count();
	if ((int)opt->param_lr.size() < np) opt->param_lr.resize(np, -1.0);
	for (int i = 0; i < np; i++) {
		if (param_is_buffer(i)) continue; /* buffers have no LR — never stepped */
		if (strcmp(param_name(i), name) == 0) {
			opt->param_lr[i] = lr;
			return;
		}
	}
}

extern "C" void optimizer_own_param(OptimizerHandle h, const char* name) {
	((Optimizer*)h)->owned.insert(name);
}

extern "C" void optimizer_set_lr(OptimizerHandle h, double lr) {
	auto* opt = (Optimizer*)h;
	opt->lr = lr;
}

static bool mlx_opt_compile_enabled(void) {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("MLX_OPT_COMPILE");
		cached = ((e != nullptr) && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

/* Adam step compiled via mx::compile. Layout of inputs vector:
     [0 .. N-1]            params (current values)
     [N .. 2N-1]            grads
     [2N .. 3N-1]           m buffers (exp_avg)
     [3N .. 4N-1]           v buffers (exp_avg_sq)
     [4N .. 5N-1]           per-param learning rates (scalar arrays)
     [5N + 0 .. 5N + 6]     beta1, 1-beta1, beta2, 1-beta2, bc1, bc2, eps
   Outputs:
     [0 .. N-1]             new params
     [N .. 2N-1]            new m
     [2N .. 3N-1]           new v

   The compiled function is cached in a map keyed on N (active param count).
   mlx caches by input-shape signature internally, so repeated calls with the
   same shape tuple hit the trace cache after first invocation. Recreating
   the lambda per call would miss mlx's identity-based cache. */
static std::unordered_map<int, std::function<std::vector<mx::array>(const std::vector<mx::array>&)>>
    adam_compiled_by_n;

static std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
get_adam_compiled(int n) {
	auto it = adam_compiled_by_n.find(n);
	if (it != adam_compiled_by_n.end()) return it->second;
	auto raw = [n](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
		const mx::array& beta1 = ins[5 * n + 0];
		const mx::array& one_b1 = ins[5 * n + 1];
		const mx::array& beta2 = ins[5 * n + 2];
		const mx::array& one_b2 = ins[5 * n + 3];
		const mx::array& bc1 = ins[5 * n + 4];
		const mx::array& bc2 = ins[5 * n + 5];
		const mx::array& eps = ins[5 * n + 6];
		std::vector<mx::array> new_p, new_m, new_v;
		new_p.reserve(n);
		new_m.reserve(n);
		new_v.reserve(n);
		for (int i = 0; i < n; i++) {
			const mx::array& p = ins[i];
			const mx::array& g = ins[n + i];
			const mx::array& m = ins[2 * n + i];
			const mx::array& v = ins[3 * n + i];
			const mx::array& lr = ins[4 * n + i];
			auto m_n = mx::add(mx::multiply(beta1, m), mx::multiply(one_b1, g));
			auto v_n = mx::add(mx::multiply(beta2, v), mx::multiply(one_b2, mx::square(g)));
			auto mhat = mx::divide(m_n, bc1);
			auto vhat = mx::divide(v_n, bc2);
			auto p_n =
			    mx::subtract(p, mx::divide(mx::multiply(lr, mhat), mx::add(mx::sqrt(vhat), eps)));
			new_p.push_back(p_n);
			new_m.push_back(m_n);
			new_v.push_back(v_n);
		}
		std::vector<mx::array> outs;
		outs.reserve(static_cast<std::size_t>(n) * 3);
		for (auto& a : new_p)
			outs.push_back(a);
		for (auto& a : new_m)
			outs.push_back(a);
		for (auto& a : new_v)
			outs.push_back(a);
		return outs;
	};
	adam_compiled_by_n[n] = mx::compile(raw);
	return adam_compiled_by_n[n];
}

static void adam_step_compile(Optimizer* opt, int np) {
	/* Gather active params (must have grads) and corresponding state. */
	std::vector<int> active_idx;
	active_idx.reserve(np);
	for (int i = 0; i < np; i++) {
		if ((opt != nullptr) && !opt_owns_param_mlx(opt, i)) continue;
		if (param_is_buffer(i)) continue; /* non-learnable buffer — never stepped */
		auto* t = (Tensor*)param_tensor(i);
		if (!t->has_grad) continue;
		active_idx.push_back(i);
	}
	int const n = (int)active_idx.size();
	if (n == 0) return;

	/* Build the input vector: params, grads, m, v, lrs, hparams. */
	auto* t0 = (Tensor*)param_tensor(active_idx[0]);
	mx::Dtype const dt = t0->data.dtype();
	std::vector<mx::array> ins;
	ins.reserve(5 * n + 7);
	for (int j = 0; j < n; j++) {
		auto* t = (Tensor*)param_tensor(active_idx[j]);
		ins.push_back(t->data);
	}
	for (int j = 0; j < n; j++) {
		auto* t = (Tensor*)param_tensor(active_idx[j]);
		ins.push_back(t->grad);
	}
	for (int j = 0; j < n; j++)
		ins.push_back(opt->m_bufs[active_idx[j]]);
	for (int j = 0; j < n; j++)
		ins.push_back(opt->v_bufs[active_idx[j]]);
	for (int j = 0; j < n; j++) {
		int const i = active_idx[j];
		double lr = opt->lr;
		if (i < (int)opt->param_lr.size() && opt->param_lr[i] >= 0) lr = opt->param_lr[i];
		ins.push_back(mx::array(lr, dt));
	}
	ins.push_back(mx::array(opt->beta1, dt));
	ins.push_back(mx::array(1.0 - opt->beta1, dt));
	ins.push_back(mx::array(opt->beta2, dt));
	ins.push_back(mx::array(1.0 - opt->beta2, dt));
	ins.push_back(mx::array(1.0 - std::pow(opt->beta1, opt->t), dt));
	ins.push_back(mx::array(1.0 - std::pow(opt->beta2, opt->t), dt));
	ins.push_back(mx::array(opt->eps, dt));

	auto& fn = get_adam_compiled(n);
	extern int g_compile_invocations;
	g_compile_invocations++;
	auto outs = fn(ins);

	/* Scatter outputs back into params / m / v. */
	for (int j = 0; j < n; j++) {
		int const i = active_idx[j];
		auto* t = (Tensor*)param_tensor(i);
		t->data = outs[j];
		opt->m_bufs[i] = outs[n + j];
		opt->v_bufs[i] = outs[2 * n + j];
	}
}

extern "C" void optimizer_step(OptimizerHandle h) {
	double const t0_opt = _wall_ms_mlx();
	auto* opt = (Optimizer*)h;
	opt->t++;
	int const np = param_count();
	_dbg_dump_param_grads_if_enabled_mlx();

	// Ensure optimizer buffers
	if ((int)opt->m_bufs.size() != np) {
		opt->m_bufs.clear();
		opt->v_bufs.clear();
		for (int i_ = 0; i_ < param_count(); i_++) {
			auto* p_tensor = (Tensor*)param_tensor(i_);
			opt->m_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
			opt->v_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
		}
	}

	/* Adam-only compile path: gate via MLX_OPT_COMPILE=1. */
	if (opt->type == 2 && mlx_opt_compile_enabled()) {
		double const tm0 = _wall_ms_mlx();
		adam_step_compile(opt, np);
		prof_optimizer_math_ms_mlx += _wall_ms_mlx() - tm0;
		std::vector<mx::array> to_eval;
		to_eval.reserve(param_count());
		for (int i_ = 0; i_ < param_count(); i_++) {
			to_eval.push_back(((Tensor*)param_tensor(i_))->data);
		}
		mx::eval(to_eval);
		tape_reset();
		for (int i_ = 0; i_ < param_count(); i_++) {
			auto* p_tensor = (Tensor*)param_tensor(i_);
			p_tensor->tape_idx = -1;
			p_tensor->has_grad = false;
			tape_append(OP_CONST, p_tensor, nullptr, nullptr, 0);
		}
		prof_optimizer_ms_mlx += _wall_ms_mlx() - t0_opt;
		prof_epochs_mlx++;
		return;
	}

	double const tm0 = _wall_ms_mlx();

	/* Hoist optimizer-state scalars out of the per-param loop. Build them in
	   the dtype of the first eligible param so they don't force a runtime
	   promotion at every multiply. Mixed-dtype models would rely on mlx's
	   promotion rules at the multiply boundary; split the optimizer by
	   dtype if that's a problem. */
	mx::Dtype opt_dtype = mx::float32;
	for (int i = 0; i < np; i++) {
		auto* p_tensor = (Tensor*)param_tensor(i);
		if (opt_owns_param_mlx(opt, i) && p_tensor->has_grad) {
			opt_dtype = p_tensor->data.dtype();
			break;
		}
	}
	auto alpha_arr = mx::array(opt->alpha, opt_dtype);
	auto one_m_alpha = mx::array(1.0 - opt->alpha, opt_dtype);
	auto beta1_arr = mx::array(opt->beta1, opt_dtype);
	auto one_m_beta1 = mx::array(1.0 - opt->beta1, opt_dtype);
	auto beta2_arr = mx::array(opt->beta2, opt_dtype);
	auto one_m_beta2 = mx::array(1.0 - opt->beta2, opt_dtype);
	auto eps_arr = mx::array(opt->eps, opt_dtype);
	auto momentum_a = mx::array(opt->momentum, opt_dtype);
	auto bc1_arr = mx::array(1.0 - std::pow(opt->beta1, opt->t), opt_dtype);
	auto bc2_arr = mx::array(1.0 - std::pow(opt->beta2, opt->t), opt_dtype);

	for (int i = 0; i < np; i++) {
		if ((opt != nullptr) && !opt_owns_param_mlx(opt, i)) continue;
		if (param_is_buffer(i)) continue; /* non-learnable buffer — never stepped */
		auto* t = (Tensor*)param_tensor(i);
		if (!t->has_grad) continue;

		/* Don't eval(t->grad) here — that's a per-param sync. The ops below
		   take lazy mx::array inputs happily; the trailing mx::eval(to_eval)
		   past the loop walks the dependency graph and pulls grads into the
		   same batch as the param updates. */
		auto g = t->grad;

		double lr = opt->lr;
		if (i < (int)opt->param_lr.size() && opt->param_lr[i] >= 0) lr = opt->param_lr[i];
		auto lr_arr = scalar_like(lr, t->data);

		switch (opt->type) {
		case 0: // SGD
			t->data = mx::subtract(t->data, mx::multiply(lr_arr, g));
			break;
		case 1: { // RMSprop — keep lr OUTSIDE the momentum buffer to match
			      // torch.optim.RMSprop. Folding lr into the buffer coincides
			      // with PyTorch only at constant lr; under an LR schedule the
			      // buffer carries stale rates and diverges.
			opt->v_bufs[i] = mx::add(mx::multiply(alpha_arr, opt->v_bufs[i]),
			                         mx::multiply(one_m_alpha, mx::square(g)));
			auto avg = mx::add(mx::sqrt(opt->v_bufs[i]), eps_arr);
			if (opt->momentum > 0) {
				opt->m_bufs[i] =
				    mx::add(mx::multiply(momentum_a, opt->m_bufs[i]), mx::divide(g, avg));
				t->data = mx::subtract(t->data, mx::multiply(lr_arr, opt->m_bufs[i]));
			} else {
				t->data = mx::subtract(t->data, mx::divide(mx::multiply(lr_arr, g), avg));
			}
			break;
		}
		case 2: { // Adam
			opt->m_bufs[i] =
			    mx::add(mx::multiply(beta1_arr, opt->m_bufs[i]), mx::multiply(one_m_beta1, g));
			opt->v_bufs[i] = mx::add(mx::multiply(beta2_arr, opt->v_bufs[i]),
			                         mx::multiply(one_m_beta2, mx::square(g)));
			auto mhat = mx::divide(opt->m_bufs[i], bc1_arr);
			auto vhat = mx::divide(opt->v_bufs[i], bc2_arr);
			t->data = mx::subtract(
			    t->data, mx::divide(mx::multiply(lr_arr, mhat), mx::add(mx::sqrt(vhat), eps_arr)));
			break;
		}
		case 3: { // AdamW (decoupled weight decay, PRE-step — torch.optim.AdamW order)
			opt->m_bufs[i] =
			    mx::add(mx::multiply(beta1_arr, opt->m_bufs[i]), mx::multiply(one_m_beta1, g));
			opt->v_bufs[i] = mx::add(mx::multiply(beta2_arr, opt->v_bufs[i]),
			                         mx::multiply(one_m_beta2, mx::square(g)));
			auto mhat = mx::divide(opt->m_bufs[i], bc1_arr);
			auto vhat = mx::divide(opt->v_bufs[i], bc2_arr);
			/* Decoupled weight decay on the PRE-step weight, then the Adam
			   update (param.mul_(1 - lr*wd) before addcdiv_). */
			t->data = mx::subtract(
			    t->data, mx::multiply(scalar_like(lr * opt->weight_decay, t->data), t->data));
			t->data = mx::subtract(
			    t->data, mx::divide(mx::multiply(lr_arr, mhat), mx::add(mx::sqrt(vhat), eps_arr)));
			break;
		}
		// GCOVR_EXCL_START — unreachable default: opt->type is always 0-3 from the create_* ctors;
		// no in-tree path sets an out-of-range type on mlx
		default:
			break;
			// GCOVR_EXCL_STOP
		}
	}
	prof_optimizer_math_ms_mlx += _wall_ms_mlx() - tm0;

	// Eval all updated params
	std::vector<mx::array> to_eval;
	to_eval.reserve(param_count());
	for (int i_ = 0; i_ < param_count(); i_++) {
		to_eval.push_back(((Tensor*)param_tensor(i_))->data);
	}
	mx::eval(to_eval);

	// Reset tape
	tape_reset();
	for (int i_ = 0; i_ < param_count(); i_++) {
		auto* p_tensor = (Tensor*)param_tensor(i_);
		p_tensor->tape_idx = -1;
		p_tensor->has_grad = false;
		tape_append(OP_CONST, p_tensor, nullptr, nullptr, 0);
	}
	prof_optimizer_ms_mlx += _wall_ms_mlx() - t0_opt;
	prof_epochs_mlx++;
}

/* Internal: clip grads for params this optimizer owns (empty set = all). */
static void clip_grad_value_filtered(Optimizer* opt, double max_val) {
	for (int i = 0; i < param_count(); i++) {
		auto* p_tensor = (Tensor*)param_tensor(i);
		if (param_is_buffer(i)) continue; /* buffers contribute no grad to clip */
		if ((opt != nullptr) && !opt_owns_param_mlx(opt, i)) continue;
		if (p_tensor->has_grad) {
			auto lo = scalar_like(-max_val, p_tensor->grad);
			auto hi = scalar_like(max_val, p_tensor->grad);
			p_tensor->grad = mx::clip(p_tensor->grad, lo, hi);
		}
	}
}

static double clip_grad_norm_filtered(Optimizer* opt, double max_norm) {
	/* Compute squared-grad sum per-param in the param's own dtype, then
	   reduce to a double on the host. Avoids mixing dtypes in a single
	   running `total` array (param dtypes may differ across the registry). */
	double sumsq = 0.0;
	for (int i = 0; i < param_count(); i++) {
		auto* p_tensor = (Tensor*)param_tensor(i);
		if (param_is_buffer(i)) continue; /* buffers contribute no grad norm */
		if ((opt != nullptr) && !opt_owns_param_mlx(opt, i)) continue;
		if (p_tensor->has_grad) {
			auto s = mx::sum(mx::square(p_tensor->grad));
			mx::eval(s);
			if (s.dtype() == mx::float64)
				sumsq += s.item<double>();
			else
				sumsq += (double)s.item<float>();
		}
	}
	double const norm = std::sqrt(sumsq);
	if (norm > max_norm) {
		double const scale = max_norm / norm;
		for (int i = 0; i < param_count(); i++) {
			auto* p_tensor = (Tensor*)param_tensor(i);
			if (param_is_buffer(i)) continue; /* buffers contribute no grad norm */
			if ((opt != nullptr) && !opt_owns_param_mlx(opt, i)) continue;
			if (p_tensor->has_grad) {
				p_tensor->grad = mx::multiply(p_tensor->grad, scalar_like(scale, p_tensor->grad));
			}
		}
	}
	return norm;
}

extern "C" void optimizer_clip_grad_value(double max_val) {
	clip_grad_value_filtered(nullptr, max_val); /* nullptr => all params */
}

extern "C" double optimizer_clip_grad_norm(double max_norm) {
	return clip_grad_norm_filtered(nullptr, max_norm); /* nullptr => all params */
}

/* Polyak soft update: mirror of the tape/torch implementation. */
extern "C" int polyak_blend_pair(double tau, const char* online_name, const char* target_name) {
	if ((online_name == nullptr) || (target_name == nullptr)) return 0;
	Tensor const* on_t = nullptr;
	Tensor* tg_t = nullptr;
	/* Exact-match both names — no prefix logic, so a name that is a proper
	   prefix of another can't over-match. */
	for (int i = 0; i < param_count(); i++) {
		std::string const nm(param_name(i));
		if ((on_t == nullptr) && nm == online_name) on_t = (Tensor*)param_tensor(i);
		if ((tg_t == nullptr) && nm == target_name) tg_t = (Tensor*)param_tensor(i);
	}
	if ((on_t == nullptr) || (tg_t == nullptr)) return 0;
	if (on_t->data.shape() != tg_t->data.shape()) return 0;
	auto tau_arr = scalar_like(tau, tg_t->data);
	auto one_minus_tau = scalar_like(1.0 - tau, tg_t->data);
	tg_t->data =
	    mx::add(mx::multiply(one_minus_tau, tg_t->data), mx::multiply(tau_arr, on_t->data));
	mx::eval(tg_t->data);
	return 1;
}

/* Optimizer buffer accessors (for serialization). */

extern "C" int optimizer_buf_count(OptimizerHandle h) {
	(void)h;
	return param_count();
}

extern "C" void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
	auto* opt = (Optimizer*)h;
	if (idx >= (int)opt->m_bufs.size()) {
		int const n = (int)((Tensor*)param_tensor(idx))->data.size();
		memset(out, 0, n * sizeof(double));
		return;
	}
	mx::eval(opt->m_bufs[idx]);
	auto& arr = opt->m_bufs[idx];
	mx_to_doubles(arr, out);
}

extern "C" void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
	auto* opt = (Optimizer*)h;
	if (idx >= (int)opt->v_bufs.size()) {
		int const n = (int)((Tensor*)param_tensor(idx))->data.size();
		memset(out, 0, n * sizeof(double));
		return;
	}
	mx::eval(opt->v_bufs[idx]);
	auto& arr = opt->v_bufs[idx];
	mx_to_doubles(arr, out);
}

extern "C" void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
	auto* opt = (Optimizer*)h;
	int const np = param_count();
	if ((int)opt->m_bufs.size() != np) {
		opt->m_bufs.clear();
		opt->v_bufs.clear();
		for (int i_ = 0; i_ < param_count(); i_++) {
			auto* p_tensor = (Tensor*)param_tensor(i_);
			opt->m_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
			opt->v_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
		}
	}
	auto* t = (Tensor*)param_tensor(idx);
	opt->m_bufs[idx] = mx_array_from_doubles(data, t->data.shape(), t->data.dtype());
}

extern "C" void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
	auto* opt = (Optimizer*)h;
	int const np = param_count();
	if ((int)opt->v_bufs.size() != np) {
		opt->m_bufs.clear();
		opt->v_bufs.clear();
		for (int i_ = 0; i_ < param_count(); i_++) {
			auto* p_tensor = (Tensor*)param_tensor(i_);
			opt->m_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
			opt->v_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
		}
	}
	auto* t = (Tensor*)param_tensor(idx);
	opt->v_bufs[idx] = mx_array_from_doubles(data, t->data.shape(), t->data.dtype());
}

extern "C" void optimizer_get_meta(OptimizerHandle h, double* out9) {
	auto* opt = (Optimizer*)h;
	out9[0] = (double)opt->type;
	out9[1] = opt->lr;
	out9[2] = opt->beta1;
	out9[3] = opt->beta2;
	out9[4] = opt->eps;
	out9[5] = opt->alpha;
	out9[6] = opt->weight_decay;
	out9[7] = opt->momentum;
	out9[8] = (double)opt->t;
}

extern "C" void optimizer_set_meta(OptimizerHandle h, const double* in9) {
	auto* opt = (Optimizer*)h;
	opt->type = (int)in9[0];
	opt->lr = in9[1];
	opt->beta1 = in9[2];
	opt->beta2 = in9[3];
	opt->eps = in9[4];
	opt->alpha = in9[5];
	opt->weight_decay = in9[6];
	opt->momentum = in9[7];
	opt->t = (int)in9[8];
}

/* native_train_step + optimizer_step_with_clip reach into Optimizer for the
   prefix-scoped clip variants. They stay mlx-local for now. */
extern "C" double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                                    TensorHandle loss_ptr, double loss_val) {
	auto* o = (Optimizer*)opt;
	optimizer_zero_grad(opt);
	if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);
	if (clip_mode == 1)
		clip_grad_value_filtered(o, clip_val);
	else if (clip_mode == 2)
		clip_grad_norm_filtered(o, clip_val);
	optimizer_step(opt);
	return loss_val;
}

/* GradScaler-aware variant (A3 of #410). Mirror of torch's
   `native_train_step_scaled` but using mlx's lazy-eval array ops:
   unscale grads via `mx::multiply(g, scalar_like(inv_scale, g))`,
   check for non-finite values via `mx::all(mx::isfinite(g))` —
   forced to materialise by the host-side item-read. NaN return =
   overflow, caller halves scale and skips. */
extern "C" double native_train_step_scaled(OptimizerHandle opt, int clip_mode, double clip_val,
                                           TensorHandle loss_ptr, double loss_val, double scale) {
	auto* o = (Optimizer*)opt;
	optimizer_zero_grad(opt);
	if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);

	double const inv_scale = 1.0 / scale;
	bool has_nonfinite = false;
	for (int i = 0; i < param_count(); i++) {
		auto* p_tensor = (Tensor*)param_tensor(i);
		if (param_is_buffer(i)) continue; /* buffers carry no grad to unscale */
		if (!opt_owns_param_mlx(o, i)) continue;
		if (!p_tensor->has_grad) continue;
		p_tensor->grad = mx::multiply(p_tensor->grad, scalar_like(inv_scale, p_tensor->grad));
		auto all_fin = mx::all(mx::isfinite(p_tensor->grad));
		mx::eval(all_fin);
		if (!all_fin.item<bool>()) has_nonfinite = true;
	}
	if (has_nonfinite) return std::nan("");

	if (clip_mode == 1)
		clip_grad_value_filtered(o, clip_val);
	else if (clip_mode == 2)
		clip_grad_norm_filtered(o, clip_val);
	optimizer_step(opt);
	return loss_val * inv_scale;
}

extern "C" int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val,
                                        int dummy) {
	(void)dummy;
	auto* o = (Optimizer*)opt;
	if (clip_mode == 1)
		clip_grad_value_filtered(o, clip_val);
	else if (clip_mode == 2)
		clip_grad_norm_filtered(o, clip_val);
	optimizer_step(opt);
	optimizer_zero_grad(opt);
	return 0;
}

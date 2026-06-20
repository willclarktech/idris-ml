/* Native optimizer surface for the torch backend.
 *
 * Owns OptWrapper (the per-optimizer struct that pairs a libtorch
 * `torch::optim::Optimizer*` with our extra metadata: type tag, hparam
 * scalars, parameter-prefix scope for multi-optimizer SAC, and a
 * pending_step pulled from optimizer_set_meta on load), the four
 * fused multi-tensor *_step_foreach implementations (SGD, RMSprop,
 * Adam, AdamW), `optimizer_step` + the wrapper's lifecycle, the
 * clip-filtered helpers, polyak_blend, the optimizer-state get/set
 * accessors used by the safetensors serializer, and the two clip-
 * scoped train-step entry points (native_train_step,
 * optimizer_step_with_clip).
 *
 * The fused foreach paths replace libtorch's per-param step() with
 * batched MultiTensorApply kernels. Numerics are identical to the
 * standard formulation: m and v live in the AdamParamState slots so
 * libtorch's serializer still works through save/load. Params with
 * undefined grad are skipped in every path (matches libtorch
 * behaviour). */
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "../../backend.h"
#include "../tensor.h"
#include "intermediates.h"
#include "profiling.h"

extern "C" int param_count(void);
extern "C" void* param_tensor(int i);
extern "C" const char* param_name(int i);
extern "C" void tensor_zero_grad(TensorHandle h);
extern "C" void tensor_backward(TensorHandle h);
extern "C" int tensor_requires_grad(TensorHandle h);
extern "C" void _dbg_dump_lstm_traj_if_enabled_torch(void);

/* Helper: collect all param_registry tensors into a vector */
static std::vector<at::Tensor> collect_param_tensors() {
	std::vector<at::Tensor> params;
	params.reserve((size_t)param_count());
	for (int i_ = 0; i_ < param_count(); i_++) {
		if (param_is_buffer(i_)) continue; /* buffers are never handed to the optimizer */
		auto* tensor = (at::Tensor*)param_tensor(i_);
		params.push_back(*tensor);
	}
	return params;
}

/* Wrapper to track optimizer type alongside the PyTorch optimizer. */
struct OptWrapper {
	int type; // 0=sgd, 1=rmsprop, 2=adam, 3=adamw
	double lr, beta1, beta2, eps, alpha, weight_decay, momentum;
	torch::optim::Optimizer* opt;
	std::string prefix; // empty = manages all params; else only params whose
	                    // registry name starts with `prefix` (SAC multi-opt)
	std::unordered_map<const void*, double>
	    param_lr;             // per-param LR overrides, keyed by the param's
	                          // TensorImpl* (stable across re-sync). Empty in the
	                          // common case → optimizer_step takes the single-LR
	                          // fast path. Mirrors tape/mlx's param_lr arrays;
	                          // libtorch has no native per-param LR within a group,
	                          // so optimizer_step buckets params by effective LR.
	int64_t pending_step = 0; // step count restored by optimizer_set_meta,
	                          // stamped onto per-param state when it is first
	                          // created (lazily, in optimizer_set_m/_v) — the
	                          // step lives inside Adam/RMSprop ParamState,
	                          // which doesn't exist on a freshly-loaded opt.
};

static std::vector<at::Tensor> collect_param_tensors_filtered(const std::string& prefix) {
	std::vector<at::Tensor> params;
	params.reserve((size_t)param_count());
	for (int i_ = 0; i_ < param_count(); i_++) {
		if (param_is_buffer(i_)) continue; /* buffers contribute no grad to clip */
		auto* tensor = (at::Tensor*)param_tensor(i_);
		if (prefix.empty()) {
			params.push_back(*tensor);
		} else {
			std::string name(param_name(i_));
			if (name.rfind(prefix, 0) == 0) {
				params.push_back(*tensor);
			}
		}
	}
	return params;
}

extern "C" OptimizerHandle optimizer_create_sgd(double lr) {
	auto params = collect_param_tensors();
	auto* w = new OptWrapper();
	w->type = 0;
	w->lr = lr;
	w->opt = new torch::optim::SGD(params, torch::optim::SGDOptions(lr));
	return static_cast<OptimizerHandle>(w);
}

extern "C" OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                                    double weight_decay, double momentum) {
	auto params = collect_param_tensors();
	auto* w = new OptWrapper();
	w->type = 1;
	w->lr = lr;
	w->alpha = alpha;
	w->eps = eps;
	w->weight_decay = weight_decay;
	w->momentum = momentum;
	w->opt = new torch::optim::RMSprop(params, torch::optim::RMSpropOptions(lr)
	                                               .alpha(alpha)
	                                               .eps(eps)
	                                               .weight_decay(weight_decay)
	                                               .momentum(momentum));
	return static_cast<OptimizerHandle>(w);
}

extern "C" OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2,
                                                 double eps) {
	auto params = collect_param_tensors();
	auto* w = new OptWrapper();
	w->type = 2;
	w->lr = lr;
	w->beta1 = beta1;
	w->beta2 = beta2;
	w->eps = eps;
	w->opt = new torch::optim::Adam(
	    params, torch::optim::AdamOptions(lr).betas(std::make_tuple(beta1, beta2)).eps(eps));
	return static_cast<OptimizerHandle>(w);
}

extern "C" OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                                       double eps, const char* prefix) {
	std::string pfx = prefix ? prefix : "";
	auto params = collect_param_tensors_filtered(pfx);
	auto* w = new OptWrapper();
	w->type = 2;
	w->lr = lr;
	w->beta1 = beta1;
	w->beta2 = beta2;
	w->eps = eps;
	w->prefix = pfx;
	w->opt = new torch::optim::Adam(
	    params, torch::optim::AdamOptions(lr).betas(std::make_tuple(beta1, beta2)).eps(eps));
	return static_cast<OptimizerHandle>(w);
}

extern "C" OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                                  double weight_decay) {
	auto params = collect_param_tensors();
	auto* w = new OptWrapper();
	w->type = 3;
	w->lr = lr;
	w->beta1 = beta1;
	w->beta2 = beta2;
	w->eps = eps;
	w->opt = new torch::optim::AdamW(params, torch::optim::AdamWOptions(lr)
	                                             .betas(std::make_tuple(beta1, beta2))
	                                             .eps(eps)
	                                             .weight_decay(weight_decay));
	return static_cast<OptimizerHandle>(w);
}

extern "C" void optimizer_free(OptimizerHandle h) {
	auto* w = static_cast<OptWrapper*>(h);
	delete w->opt;
	delete w;
}

/* Core Adam foreach math: assumes caller has gathered lists, materialised
   state, bumped step, and entered a NoGradGuard. Shared by adam_step_foreach
   and adamw_step_foreach (AdamW adds decoupled weight-decay before the
   call but uses the same math thereafter). */
static void adam_core_foreach(double lr, double beta1, double beta2, double eps, int64_t new_step,
                              std::vector<at::Tensor>& params, std::vector<at::Tensor>& m_list,
                              std::vector<at::Tensor>& v_list, std::vector<at::Tensor>& g_list) {
	double bc1 = 1.0 - std::pow(beta1, (double)new_step);
	double bc2 = 1.0 - std::pow(beta2, (double)new_step);
	double bc2_sqrt = std::sqrt(bc2);
	double step_size = lr / bc1;

	/* m = β1·m + (1-β1)·g — matches libtorch's mul_().add_(g, 1-β1) order. */
	at::_foreach_mul_(m_list, beta1);
	at::_foreach_add_(m_list, g_list, 1.0 - beta1);

	/* v = β2·v + (1-β2)·g² */
	at::_foreach_mul_(v_list, beta2);
	at::_foreach_addcmul_(v_list, g_list, g_list, 1.0 - beta2);

	/* denom = sqrt(v) / sqrt(bc2) + eps */
	auto denom = at::_foreach_sqrt(v_list);
	at::_foreach_div_(denom, bc2_sqrt);
	at::_foreach_add_(denom, eps);

	/* p -= step_size · m / denom */
	at::_foreach_addcdiv_(params, m_list, denom, -step_size);
}

static void adam_step_foreach(OptWrapper* w, const std::vector<at::Tensor>& params, double lr) {
	auto& opt = *w->opt;
	auto& state = opt.state();

	std::vector<at::Tensor> active_params, m_list, v_list, g_list;
	active_params.reserve(params.size());
	m_list.reserve(params.size());
	v_list.reserve(params.size());
	g_list.reserve(params.size());

	int64_t new_step = 0;
	for (const auto& p : params) {
		if (!p.grad().defined()) continue;
		auto key = p.unsafeGetTensorImpl();
		if (state.count(key) == 0) {
			state[key] = std::make_unique<torch::optim::AdamParamState>();
			auto& s0 = static_cast<torch::optim::AdamParamState&>(*state[key]);
			s0.exp_avg(at::zeros_like(p));
			s0.exp_avg_sq(at::zeros_like(p));
			s0.step(0);
		}
		auto& s = static_cast<torch::optim::AdamParamState&>(*state.at(key));
		s.step(s.step() + 1);
		new_step = s.step();
		active_params.push_back(p);
		m_list.push_back(s.exp_avg());
		v_list.push_back(s.exp_avg_sq());
		g_list.push_back(p.grad());
	}

	if (active_params.empty()) return;

	/* In-place updates on leaf params with requires_grad=true would trip
	   autograd's check_inplace. Same wrap as torch::optim::Adam::step(). */
	torch::NoGradGuard no_grad;
	adam_core_foreach(lr, w->beta1, w->beta2, w->eps, new_step, active_params, m_list, v_list,
	                  g_list);
}

/* AdamW: decoupled weight-decay applied to params as `p *= 1 - lr*wd`
   BEFORE the Adam math. AdamWParamState is a distinct libtorch type from
   AdamParamState but exposes the same field accessors, so the shared
   adam_core_foreach math is reusable. */
static void adamw_step_foreach(OptWrapper* w, const std::vector<at::Tensor>& params, double lr) {
	auto& opt = *w->opt;
	auto& state = opt.state();

	std::vector<at::Tensor> active_params, m_list, v_list, g_list;
	active_params.reserve(params.size());
	m_list.reserve(params.size());
	v_list.reserve(params.size());
	g_list.reserve(params.size());

	int64_t new_step = 0;
	for (const auto& p : params) {
		if (!p.grad().defined()) continue;
		auto key = p.unsafeGetTensorImpl();
		if (state.count(key) == 0) {
			state[key] = std::make_unique<torch::optim::AdamWParamState>();
			auto& s0 = static_cast<torch::optim::AdamWParamState&>(*state[key]);
			s0.exp_avg(at::zeros_like(p));
			s0.exp_avg_sq(at::zeros_like(p));
			s0.step(0);
		}
		auto& s = static_cast<torch::optim::AdamWParamState&>(*state.at(key));
		s.step(s.step() + 1);
		new_step = s.step();
		active_params.push_back(p);
		m_list.push_back(s.exp_avg());
		v_list.push_back(s.exp_avg_sq());
		g_list.push_back(p.grad());
	}

	if (active_params.empty()) return;

	torch::NoGradGuard no_grad;

	/* Decoupled weight decay: p *= 1 - lr*wd  (skip when wd == 0). Uses the
	   effective (possibly per-param-overridden) lr, matching PyTorch's
	   per-group decoupling. */
	if (w->weight_decay != 0.0) {
		at::_foreach_mul_(active_params, 1.0 - lr * w->weight_decay);
	}

	adam_core_foreach(lr, w->beta1, w->beta2, w->eps, new_step, active_params, m_list, v_list,
	                  g_list);
}

/* Fused RMSprop step (non-centered). Mirrors libtorch's
   `torch::optim::RMSprop::step()` op order:
     (optional)  g_eff = g + weight_decay * p     (fresh clone)
                 v.mul_(α).addcmul_(g, g, 1 - α)
                 avg = sqrt(v) + eps               (fresh tensor)
     (momentum)  buf.mul_(m).addcdiv_(g, avg);     p -= lr * buf
     (no momentum)                                 p -= lr * g / avg
   v / buf live in RMSpropParamState. */
static void rmsprop_step_foreach(OptWrapper* w, const std::vector<at::Tensor>& params, double lr) {
	auto& opt = *w->opt;
	auto& state = opt.state();
	const bool use_momentum = (w->momentum > 0.0);
	const bool use_wd = (w->weight_decay != 0.0);

	std::vector<at::Tensor> active_params, v_list, g_list, buf_list;
	active_params.reserve(params.size());
	v_list.reserve(params.size());
	g_list.reserve(params.size());
	if (use_momentum) buf_list.reserve(params.size());

	for (const auto& p : params) {
		if (!p.grad().defined()) continue;
		auto key = p.unsafeGetTensorImpl();
		if (state.count(key) == 0) {
			state[key] = std::make_unique<torch::optim::RMSpropParamState>();
			auto& s0 = static_cast<torch::optim::RMSpropParamState&>(*state[key]);
			s0.square_avg(at::zeros_like(p));
			s0.step(0);
		}
		auto& s = static_cast<torch::optim::RMSpropParamState&>(*state.at(key));
		s.step(s.step() + 1);
		if (use_momentum && !s.momentum_buffer().defined()) {
			s.momentum_buffer(at::zeros_like(p));
		}
		active_params.push_back(p);
		v_list.push_back(s.square_avg());
		g_list.push_back(p.grad());
		if (use_momentum) buf_list.push_back(s.momentum_buffer());
	}

	if (active_params.empty()) return;

	torch::NoGradGuard no_grad;

	std::vector<at::Tensor> g_eff;
	if (use_wd) {
		g_eff.reserve(g_list.size());
		for (auto& g : g_list)
			g_eff.push_back(g.clone());
		at::_foreach_add_(g_eff, active_params, w->weight_decay);
	} else {
		g_eff = g_list;
	}

	double alpha = w->alpha, eps = w->eps;
	at::_foreach_mul_(v_list, alpha);
	at::_foreach_addcmul_(v_list, g_eff, g_eff, 1.0 - alpha);
	auto avg = at::_foreach_sqrt(v_list);
	at::_foreach_add_(avg, eps);

	if (use_momentum) {
		at::_foreach_mul_(buf_list, w->momentum);
		at::_foreach_addcdiv_(buf_list, g_eff, avg, 1.0);
		at::_foreach_add_(active_params, buf_list, -lr);
	} else {
		at::_foreach_addcdiv_(active_params, g_eff, avg, -lr);
	}
}

/* SGD: our wrapper exposes only lr (no momentum / wd / nesterov), so the
   math collapses to a single _foreach_add_ call. */
static void sgd_step_foreach(OptWrapper* w, const std::vector<at::Tensor>& params, double lr) {
	(void)w;
	std::vector<at::Tensor> active, grads;
	active.reserve(params.size());
	grads.reserve(params.size());
	for (const auto& p : params) {
		if (!p.grad().defined()) continue;
		active.push_back(p);
		grads.push_back(p.grad());
	}
	if (active.empty()) return;
	torch::NoGradGuard no_grad;
	at::_foreach_add_(active, grads, -lr);
}

/* Dispatch one effective-LR bucket to the matching fused step. */
static void dispatch_step_foreach(OptWrapper* w, const std::vector<at::Tensor>& params, double lr) {
	switch (w->type) {
	case 0:
		sgd_step_foreach(w, params, lr);
		break;
	case 1:
		rmsprop_step_foreach(w, params, lr);
		break;
	case 2:
		adam_step_foreach(w, params, lr);
		break;
	case 3:
		adamw_step_foreach(w, params, lr);
		break;
	default:
		w->opt->step();
	}
}

extern "C" void optimizer_step(OptimizerHandle h) {
	double t0 = _wall_ms_torch();
	auto* w = static_cast<OptWrapper*>(h);
	auto* opt = w->opt;
	/* Re-sync param list from registry (handles late registration via autoName).
	   For group-scoped optimizers, only sync params whose name starts with w->prefix. */
	auto& param_groups = opt->param_groups();
	if (!param_groups.empty()) {
		auto& params_ref = param_groups[0].params();
		auto current = collect_param_tensors_filtered(w->prefix);
		if (params_ref.size() != current.size()) {
			params_ref.clear();
			for (auto& t : current)
				params_ref.push_back(t);
		}
		double tm0 = _wall_ms_torch();
		/* TORCH_FOREACH=0 disables every fused multi-tensor path for A/B
		   perf comparison. Defaults to on. */
		static const bool foreach_enabled = []() {
			const char* e = std::getenv("TORCH_FOREACH");
			return !(e && (e[0] == '0'));
		}();
		if (foreach_enabled) {
			if (w->param_lr.empty()) {
				/* Common case: one LR for every param — single fused pass. */
				dispatch_step_foreach(w, params_ref, w->lr);
			} else {
				/* Per-param LR overrides active (restrictTo / freezeGroup /
				   setParamLR). libtorch has no per-param LR within a param
				   group, so bucket params by effective LR and run one fused
				   pass per bucket. Each param appears in exactly one bucket,
				   so its state (m/v/step) is touched exactly once. Buckets are
				   rebuilt every step from the override map keyed on TensorImpl*,
				   so an overridden param keeps its LR across the registry
				   re-sync above (no group membership to preserve). */
				std::unordered_map<double, std::vector<at::Tensor>> buckets;
				for (auto& p : params_ref) {
					double eff_lr = w->lr;
					auto it = w->param_lr.find(p.unsafeGetTensorImpl());
					if (it != w->param_lr.end()) eff_lr = it->second;
					buckets[eff_lr].push_back(p);
				}
				for (auto& kv : buckets)
					dispatch_step_foreach(w, kv.second, kv.first);
			}
		} else {
			opt->step();
		}
		prof_optimizer_math_ms_torch += _wall_ms_torch() - tm0;
	} else {
		double tm0 = _wall_ms_torch();
		opt->step();
		prof_optimizer_math_ms_torch += _wall_ms_torch() - tm0;
	}
	_dbg_dump_lstm_traj_if_enabled_torch();
	free_intermediates();
	prof_optimizer_ms_torch += _wall_ms_torch() - t0;
	prof_epochs_torch++;
}

extern "C" void optimizer_zero_grad(OptimizerHandle h) {
	static_cast<OptWrapper*>(h)->opt->zero_grad();
}

extern "C" void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
	/* Record a per-param LR override keyed on the param's TensorImpl* (stable
	   across the registry re-sync in optimizer_step). The override is consumed
	   by optimizer_step's per-LR bucketing — libtorch has no native per-param
	   LR within a param group. Mirrors tape/mlx's name→lr lookup. lr < 0 (the
	   -1 sentinel) clears any existing override, restoring the base LR. */
	auto* w = static_cast<OptWrapper*>(h);
	for (int i = 0; i < param_count(); i++) {
		if (param_is_buffer(i)) continue; /* buffers have no LR — never stepped */
		if (std::strcmp(param_name(i), name) == 0) {
			const void* key = ((at::Tensor*)param_tensor(i))->unsafeGetTensorImpl();
			if (lr < 0.0)
				w->param_lr.erase(key);
			else
				w->param_lr[key] = lr;
			return;
		}
	}
}

extern "C" void optimizer_set_lr(OptimizerHandle h, double lr) {
	auto* w = static_cast<OptWrapper*>(h);
	w->lr = lr;
	/* Update the LR on each param group's options. */
	for (auto& g : w->opt->param_groups()) {
		switch (w->type) {
		case 0:
			static_cast<torch::optim::SGDOptions&>(g.options()).lr(lr);
			break;
		case 1:
			static_cast<torch::optim::RMSpropOptions&>(g.options()).lr(lr);
			break;
		case 2:
			static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
			break;
		case 3:
			static_cast<torch::optim::AdamWOptions&>(g.options()).lr(lr);
			break;
		}
	}
}

static void clip_grad_value_filtered(const std::string& prefix, double max_val) {
	auto params = collect_param_tensors_filtered(prefix);
	torch::nn::utils::clip_grad_value_(params, max_val);
}

static double clip_grad_norm_filtered(const std::string& prefix, double max_norm) {
	auto params = collect_param_tensors_filtered(prefix);
	return torch::nn::utils::clip_grad_norm_(params, max_norm);
}

extern "C" void optimizer_clip_grad_value(double max_val) {
	clip_grad_value_filtered("", max_val);
}

extern "C" double optimizer_clip_grad_norm(double max_norm) {
	return clip_grad_norm_filtered("", max_norm);
}

/* Polyak soft update: mirror of the tape-backend implementation. */
extern "C" int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
	if (!online_scope || !target_scope) return 0;
	std::string on_s(online_scope), tg_s(target_scope);
	int blended = 0;
	torch::NoGradGuard no_grad;
	for (int i = 0; i < param_count(); i++) {
		if (param_is_buffer(i)) continue; /* buffers aren't part of the EMA */
		std::string on_name(param_name(i));
		if (on_name.rfind(on_s, 0) != 0) continue;
		std::string tgt_name = tg_s + on_name.substr(on_s.size());
		for (int j = 0; j < param_count(); j++) {
			if (std::string(param_name(j)) != tgt_name) continue;
			at::Tensor& on_t = *(at::Tensor*)param_tensor(i);
			at::Tensor& tg_t = *(at::Tensor*)param_tensor(j);
			if (!on_t.sizes().equals(tg_t.sizes())) break;
			tg_t.mul_(1.0 - tau).add_(on_t, tau);
			blended++;
			break;
		}
	}
	return blended;
}

/* Optimizer buffer accessors (used by safetensors serializer). */

static void* param_state_key(torch::optim::Optimizer* opt, int idx) {
	auto& params = opt->param_groups()[0].params();
	if (idx >= (int)params.size()) return nullptr;
	return params[idx].unsafeGetTensorImpl();
}

extern "C" int optimizer_buf_count(OptimizerHandle h) {
	(void)h;
	return (int)param_count();
}

extern "C" void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
	auto* w = static_cast<OptWrapper*>(h);
	int numel = (int)((at::Tensor*)param_tensor(idx))->numel();
	auto key = param_state_key(w->opt, idx);
	if (!key || w->opt->state().count(key) == 0) {
		memset(out, 0, numel * sizeof(double));
		return;
	}
	auto& state = *w->opt->state().at(key);
	at::Tensor buf;
	if (w->type == 2) { /* Adam */
		buf = static_cast<torch::optim::AdamParamState&>(state).exp_avg();
	} else if (w->type == 1) { /* RMSprop */
		auto& rms = static_cast<torch::optim::RMSpropParamState&>(state);
		buf = rms.momentum_buffer().defined() ? rms.momentum_buffer()
		                                      : at::zeros_like(*(at::Tensor*)param_tensor(idx));
	} else {
		memset(out, 0, numel * sizeof(double));
		return;
	}
	buf = buf.cpu().contiguous().to(torch::kFloat64);
	memcpy(out, buf.data_ptr<double>(), numel * sizeof(double));
}

extern "C" void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
	auto* w = static_cast<OptWrapper*>(h);
	int numel = (int)((at::Tensor*)param_tensor(idx))->numel();
	auto key = param_state_key(w->opt, idx);
	if (!key || w->opt->state().count(key) == 0) {
		memset(out, 0, numel * sizeof(double));
		return;
	}
	auto& state = *w->opt->state().at(key);
	at::Tensor buf;
	if (w->type == 2) { /* Adam */
		buf = static_cast<torch::optim::AdamParamState&>(state).exp_avg_sq();
	} else if (w->type == 1) { /* RMSprop */
		buf = static_cast<torch::optim::RMSpropParamState&>(state).square_avg();
	} else {
		memset(out, 0, numel * sizeof(double));
		return;
	}
	buf = buf.cpu().contiguous().to(torch::kFloat64);
	memcpy(out, buf.data_ptr<double>(), numel * sizeof(double));
}

extern "C" void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
	auto* w = static_cast<OptWrapper*>(h);
	auto* param_t = (at::Tensor*)param_tensor(idx);
	int numel = (int)param_t->numel();
	auto key = param_state_key(w->opt, idx);
	if (!key) return;
	auto tensor = torch::from_blob((void*)data, {(int64_t)numel}, torch::kFloat64).clone();
	tensor = tensor.reshape(param_t->sizes());
	if (w->opt->state().count(key) == 0) {
		if (w->type == 2) {
			auto st = std::make_unique<torch::optim::AdamParamState>();
			st->step(w->pending_step);
			w->opt->state()[key] = std::move(st);
		} else if (w->type == 1) {
			auto st = std::make_unique<torch::optim::RMSpropParamState>();
			st->step(w->pending_step);
			w->opt->state()[key] = std::move(st);
		} else
			return;
	}
	auto& state = *w->opt->state().at(key);
	if (w->type == 2) {
		static_cast<torch::optim::AdamParamState&>(state).exp_avg(tensor);
	} else if (w->type == 1) {
		static_cast<torch::optim::RMSpropParamState&>(state).momentum_buffer(tensor);
	}
}

extern "C" void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
	auto* w = static_cast<OptWrapper*>(h);
	auto* param_t = (at::Tensor*)param_tensor(idx);
	int numel = (int)param_t->numel();
	auto key = param_state_key(w->opt, idx);
	if (!key) return;
	auto tensor = torch::from_blob((void*)data, {(int64_t)numel}, torch::kFloat64).clone();
	tensor = tensor.reshape(param_t->sizes());
	if (w->opt->state().count(key) == 0) {
		if (w->type == 2) {
			auto st = std::make_unique<torch::optim::AdamParamState>();
			st->step(w->pending_step);
			w->opt->state()[key] = std::move(st);
		} else if (w->type == 1) {
			auto st = std::make_unique<torch::optim::RMSpropParamState>();
			st->step(w->pending_step);
			w->opt->state()[key] = std::move(st);
		} else
			return;
	}
	auto& state = *w->opt->state().at(key);
	if (w->type == 2) {
		static_cast<torch::optim::AdamParamState&>(state).exp_avg_sq(tensor);
	} else if (w->type == 1) {
		static_cast<torch::optim::RMSpropParamState&>(state).square_avg(tensor);
	}
}

extern "C" void optimizer_get_meta(OptimizerHandle h, double* out9) {
	auto* w = static_cast<OptWrapper*>(h);
	out9[0] = (double)w->type;
	out9[1] = w->lr;
	out9[2] = w->beta1;
	out9[3] = w->beta2;
	out9[4] = w->eps;
	out9[5] = w->alpha;
	out9[6] = w->weight_decay;
	out9[7] = w->momentum;
	/* Get step count from first param's state if available */
	int64_t step = 0;
	if (!w->opt->param_groups().empty()) {
		auto& params = w->opt->param_groups()[0].params();
		if (!params.empty()) {
			auto key = params[0].unsafeGetTensorImpl();
			if (w->opt->state().count(key)) {
				auto& state = *w->opt->state().at(key);
				if (w->type == 2)
					step = static_cast<torch::optim::AdamParamState&>(state).step();
				else if (w->type == 1)
					step = static_cast<torch::optim::RMSpropParamState&>(state).step();
			}
		}
	}
	out9[8] = (double)step;
}

extern "C" void optimizer_set_meta(OptimizerHandle h, const double* in9) {
	auto* w = static_cast<OptWrapper*>(h);
	w->type = (int)in9[0];
	w->lr = in9[1];
	w->beta1 = in9[2];
	w->beta2 = in9[3];
	w->eps = in9[4];
	w->alpha = in9[5];
	w->weight_decay = in9[6];
	w->momentum = in9[7];
	/* Step count: set on all existing param states, and stash for any
	   states created later (optimizer_set_m/_v during load run after this). */
	int64_t step = (int64_t)in9[8];
	w->pending_step = step;
	if (!w->opt->param_groups().empty()) {
		for (auto& p : w->opt->param_groups()[0].params()) {
			auto key = p.unsafeGetTensorImpl();
			if (w->opt->state().count(key)) {
				auto& state = *w->opt->state().at(key);
				if (w->type == 2)
					static_cast<torch::optim::AdamParamState&>(state).step(step);
				else if (w->type == 1)
					static_cast<torch::optim::RMSpropParamState&>(state).step(step);
			}
		}
	}
}

/* native_train_step + optimizer_step_with_clip reach into OptWrapper for the
   prefix-scoped clip variants used by SAC's multi-optimizer training. They
   stay torch-local for now (Phase 7's lift was for the shared trampolines
   when both backends share the port shape). */
extern "C" double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                                    TensorHandle loss_ptr, double loss_val) {
	auto* w = static_cast<OptWrapper*>(opt);
	optimizer_zero_grad(opt);
	if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);
	if (clip_mode == 1)
		clip_grad_value_filtered(w->prefix, clip_val);
	else if (clip_mode == 2)
		clip_grad_norm_filtered(w->prefix, clip_val);
	optimizer_step(opt);
	return loss_val;
}

/* GradScaler-aware variant (A3 of #410). The caller has pre-multiplied
   the loss by `scale` so backward produces grads at the scaled
   magnitude. This op runs zero_grad + backward, then walks the prefix-
   filtered param set, divides each .grad() by scale in-place via
   libtorch's `div_`, and checks for any non-finite values via
   `at::isfinite(...).all()`. If overflow is detected, returns NaN and
   skips the step — caller halves the scale. Otherwise: clip + step +
   return unscaled loss = loss_val / scale.

   Mirror of the shared/training/optimizer.c implementation used by the
   tape backend; the only difference is that torch walks libtorch
   tensors directly (param.grad() returns an at::Tensor) instead of
   the port's data_read/grad_read accessors. */
extern "C" double native_train_step_scaled(OptimizerHandle opt, int clip_mode, double clip_val,
                                           TensorHandle loss_ptr, double loss_val, double scale) {
	auto* w = static_cast<OptWrapper*>(opt);
	optimizer_zero_grad(opt);
	if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);

	auto params = collect_param_tensors_filtered(w->prefix);
	double inv_scale = 1.0 / scale;
	bool has_nonfinite = false;
	for (auto& p : params) {
		auto g = p.grad();
		if (!g.defined()) continue;
		g.mul_(inv_scale);
		if (!at::isfinite(g).all().item<bool>()) has_nonfinite = true;
	}
	if (has_nonfinite) return std::nan("");

	if (clip_mode == 1)
		clip_grad_value_filtered(w->prefix, clip_val);
	else if (clip_mode == 2)
		clip_grad_norm_filtered(w->prefix, clip_val);
	optimizer_step(opt);
	return loss_val * inv_scale;
}

extern "C" int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val,
                                        int dummy) {
	(void)dummy;
	auto* w = static_cast<OptWrapper*>(opt);
	if (clip_mode == 1)
		clip_grad_value_filtered(w->prefix, clip_val);
	else if (clip_mode == 2)
		clip_grad_norm_filtered(w->prefix, clip_val);
	optimizer_step(opt);
	optimizer_zero_grad(opt);
	return 0;
}

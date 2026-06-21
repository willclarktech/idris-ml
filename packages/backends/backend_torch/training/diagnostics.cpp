/* Debug diagnostics for the torch backend.
 *
 * DEBUG_LSTM_TRAJ: dump h0/c0 param value trajectories. Mirrors the
 * tape backend's diagnostic so a cross-backend convergence regression
 * on RNN init can be localized. Walks the shared param registry.
 *
 * DEBUG_PARAM_GRADS: dump per-param gradient L2 norms after backward.
 * Mirrors the tape diagnostic for cross-backend comparison.
 *
 * Both are no-ops unless the corresponding env var is set. */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include "../tensor.h"

extern "C" int param_count(void);
extern "C" const char* param_name(int i);
extern "C" void* param_tensor(int i);

static int _dbg_traj_step_torch = 0;

extern "C" void _dbg_dump_lstm_traj_if_enabled_torch(void) {
	if (getenv("DEBUG_LSTM_TRAJ") == nullptr) return;
	int every = 100;
	const char* every_s = getenv("DEBUG_LSTM_TRAJ_EVERY");
	if (every_s != nullptr) every = static_cast<int>(strtol(every_s, nullptr, 10));
	every = std::max(every, 1); // env-supplied; guard the modulo below against % 0
	_dbg_traj_step_torch++;
	if (_dbg_traj_step_torch % every != 0 && _dbg_traj_step_torch != 1) return;
	for (int i = 0; i < param_count(); i++) {
		const std::string nm(param_name(i));
		if (nm.size() >= 3 &&
		    (nm.substr(nm.size() - 3) == "_h0" || nm.substr(nm.size() - 3) == "_c0")) {
			auto& t = *(at::Tensor*)param_tensor(i);
			auto t_cpu = t.detach().cpu().to(torch::kFloat64).contiguous();
			const double* d = t_cpu.data_ptr<double>();
			const int numel = (int)t.numel();
			double l2 = 0.0, mn = 1e300, mx = -1e300;
			for (int j = 0; j < numel; j++) {
				const double v = d[j];
				l2 += v * v;
				mn = std::min(mn, v);
				mx = std::max(mx, v);
			}
			l2 = std::sqrt(l2);
			fprintf(
			    stderr,
			    "[traj epoch %d] %s l2=%.10g min=%.10g max=%.10g | t[0..2]=%.10g, %.10g, %.10g\n",
			    _dbg_traj_step_torch, nm.c_str(), l2, mn, mx, numel >= 1 ? d[0] : 0.0,
			    numel >= 2 ? d[1] : 0.0, numel >= 3 ? d[2] : 0.0);
		}
	}
}

extern "C" void _dbg_dump_param_grads_if_enabled_torch(void) {
	if (getenv("DEBUG_PARAM_GRADS") == nullptr) return;
	fprintf(stderr, "=== param grads after backward (torch) ===\n");
	for (int i = 0; i < param_count(); i++) {
		const std::string name(param_name(i));
		auto* tensor = (at::Tensor*)param_tensor(i);
		double l2 = 0.0;
		int has_nan = 0;
		const int numel = (int)tensor->numel();
		if (tensor->grad().defined()) {
			auto g_cpu = tensor->grad().cpu().to(torch::kFloat64).contiguous();
			const double* g = g_cpu.data_ptr<double>();
			for (int j = 0; j < numel; j++) {
				const double v = g[j];
				if (std::isnan(v) || std::isinf(v)) has_nan = 1;
				l2 += v * v;
			}
			l2 = std::sqrt(l2);
			fprintf(stderr, "  %-40s numel=%-6d l2=%12.6e%s\n", name.c_str(), numel, l2,
			        has_nan != 0 ? " NAN_OR_INF!" : "");
		} else {
			fprintf(stderr, "  %-40s numel=%-6d NO_GRAD\n", name.c_str(), numel);
		}
	}
}

/* Debug diagnostics for the mlx backend.
 *
 * DEBUG_PARAM_GRADS: dump per-param gradient L2 norms after a backward
 * pass. Mirrors tape/torch backend diagnostics so cross-backend
 * regression on RNN init / convergence can be localized.
 *
 * Capped at DEBUG_PARAM_GRADS_MAX dumps (default 1) so a noisy training
 * loop doesn't spam stderr. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../tensor.h"
#include "../precision.h" /* mx_to_doubles */

extern "C" int param_count(void);
extern "C" const char* param_name(int i);
extern "C" void* param_tensor(int i);

extern "C" void _dbg_dump_param_grads_if_enabled_mlx(void) {
	static int dumped = 0;
	static int max_dumps = -1;
	if (max_dumps < 0) {
		const char* mx_env = getenv("DEBUG_PARAM_GRADS_MAX");
		max_dumps = (mx_env != nullptr) ? atoi(mx_env) : 1;
	}
	const char* env = getenv("DEBUG_PARAM_GRADS");
	if ((env == nullptr) || env[0] != '1') return;
	if (dumped >= max_dumps) return;
	dumped++;
	fprintf(stderr, "[DEBUG_PARAM_GRADS_MLX] dump #%d (np=%d):\n", dumped, param_count());
	for (int i = 0; i < param_count(); i++) {
		const char* p_name = param_name(i);
		auto* t = (Tensor*)param_tensor(i);
		long const n = (long)t->data.size();
		double l2 = 0.0;
		int const has_grad = t->has_grad ? 1 : 0;
		int const rg = t->requires_grad ? 1 : 0;
		if (t->has_grad) {
			mx::eval(t->grad);
			auto contig = mx::contiguous(t->grad);
			mx::eval(contig);
			std::vector<double> gbuf((size_t)n);
			mx_to_doubles(contig, gbuf.data());
			const double* gp = gbuf.data();
			for (long j = 0; j < n; j++)
				l2 += gp[j] * gp[j];
		}
		l2 = sqrt(l2);
		fprintf(stderr, "  [%d] %s (n=%ld rg=%d hg=%d) grad_l2=%.6e\n", i, p_name, n, rg, has_grad,
		        l2);
	}
	fflush(stderr);
}

/* tensor_batch_norm for the mlx backend.
 *
 * Reshape flat input to [C, spatial]. Training mode updates running
 * stats in-place via mom * mean + (1-mom) * running, then evaluates
 * eagerly — these don't participate in autograd, so the update must
 * happen before the result tape entry. Eval mode reads the already-
 * computed running stats. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_batch_norm_mlx_streamed(TensorHandle hinput, TensorHandle hgamma,
                                                       TensorHandle hbeta,
                                                       TensorHandle hrunning_mean,
                                                       TensorHandle hrunning_var, int C,
                                                       int spatial, int training, double momentum,
                                                       double eps, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	auto* gamma = (Tensor*)hgamma;
	auto* beta = (Tensor*)hbeta;
	auto* rm = (Tensor*)hrunning_mean;
	auto* rv = (Tensor*)hrunning_var;

	auto x = mx::reshape(inp->data, {C, spatial});
	auto mean = mx::mean(x, std::vector<int>{1}, true);
	auto var = mx::var(x, std::vector<int>{1}, true);

	if (training != 0) {
		auto mom = scalar_like(momentum, rm->data);
		auto one_m_mo = scalar_like(1.0 - momentum, rm->data);
		/* Bessel n/(n-1) correction on the running-var update only (matches
		 * torch::batch_norm / PyTorch); `var` stays biased for the per-batch
		 * normalization below (line ~47). */
		auto var_corrected = mx::multiply(var, scalar_like((double)spatial / (spatial - 1.0), var));
		auto new_rm =
		    mx::add(mx::multiply(one_m_mo, rm->data), mx::multiply(mom, mx::squeeze(mean)));
		auto new_rv = mx::add(mx::multiply(one_m_mo, rv->data),
		                      mx::multiply(mom, mx::squeeze(var_corrected)));
		rm->data = new_rm;
		rv->data = new_rv;
		mx::eval(rm->data);
		mx::eval(rv->data);
	} else {
		mean = mx::reshape(rm->data, {C, 1});
		var = mx::reshape(rv->data, {C, 1});
	}

	auto rstd = mx::rsqrt(mx::add(var, scalar_like(eps, var)));
	auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
	auto g = mx::reshape(gamma->data, {C, 1});
	auto b = mx::reshape(beta->data, {C, 1});
	auto result = mx::flatten(mx::add(mx::multiply(g, x_hat), b));

	bool const rg = inp->requires_grad || gamma->requires_grad || beta->requires_grad;
	auto* r = new Tensor(result, rg);
	if (rg) {
		int const idx = tape_append(OP_BATCH_NORM, r, inp, nullptr, 0);
		if (idx >= 0) {
			auto* meta = new BatchNormReplayMeta();
			meta->gamma_pool_idx = gamma->pool_idx;
			meta->beta_pool_idx = beta->pool_idx;
			meta->C = C;
			meta->spatial = spatial;
			meta->eps = eps;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma,
                                          TensorHandle hbeta, TensorHandle hrunning_mean,
                                          TensorHandle hrunning_var, int C, int spatial,
                                          int training, double momentum, double eps) {
	return tensor_batch_norm_mlx_streamed(hinput, hgamma, hbeta, hrunning_mean, hrunning_var, C,
	                                      spatial, training, momentum, eps, default_stream_tag());
}

static void mlx_replay_batch_norm(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* bm = (BatchNormReplayMeta*)e.meta;
	auto x = mx::reshape(a, {bm->C, bm->spatial});
	auto mean = mx::mean(x, std::vector<int>{1}, true);
	auto var = mx::var(x, std::vector<int>{1}, true);
	auto rstd = mx::rsqrt(mx::add(var, scalar_like(bm->eps, var)));
	auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
	auto g = mx::reshape(pool[bm->gamma_pool_idx], {bm->C, 1});
	auto bt = mx::reshape(pool[bm->beta_pool_idx], {bm->C, 1});
	pool[out] = mx::flatten(mx::add(mx::multiply(g, x_hat), bt));
}
MLX_REGISTER_REPLAY(OP_BATCH_NORM, mlx_replay_batch_norm)

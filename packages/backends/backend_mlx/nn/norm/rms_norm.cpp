/* tensor_rms_norm_2d for the mlx backend.
 *
 * Wraps `mlx::core::fast::rms_norm` — mlx-lm's canonical fused
 * implementation. Same HF LlamaRMSNorm formula as the other backends:
 *   variance_i = mean(input[i, :]^2)
 *   rstd_i     = 1 / sqrt(variance_i + eps)
 *   out[i, j]  = input[i, j] * rstd_i * weight[j]
 *
 * Replaces the per-row 7-primitive chain in
 * `HfCommon.applyRmsNorm2dRaw` with a single fused mlx node; lazy-
 * evaluated so the compute batches into mlx's regular eval boundary.
 * The replay closure recomputes via the same call during backward.
 */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

#include <mlx/fast.h>

extern "C" TensorHandle tensor_rms_norm_2d_mlx_streamed(TensorHandle h, TensorHandle hweight,
                                                        double eps, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto weight = (Tensor*)hweight;

	auto result = mx::fast::rms_norm(t->data, weight->data, (float)eps);

	bool rg = t->requires_grad || weight->requires_grad;
	auto r = new Tensor(result, rg);
	if (rg) {
		int idx = tape_append(OP_RMS_NORM_2D, r, t, nullptr, eps);
		if (idx >= 0) {
			auto meta = new RmsNormReplayMeta();
			meta->weight_pool_idx = weight->pool_idx;
			meta->eps = eps;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_rms_norm_2d(TensorHandle h, TensorHandle hweight, double eps) {
	return tensor_rms_norm_2d_mlx_streamed(h, hweight, eps, default_stream_tag());
}

static void mlx_replay_rms_norm_2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	auto meta = (RmsNormReplayMeta*)e.meta;
	auto weight = pool[meta->weight_pool_idx];
	pool[out] = mx::fast::rms_norm(a, weight, (float)meta->eps);
}
MLX_REGISTER_REPLAY(OP_RMS_NORM_2D, mlx_replay_rms_norm_2d)

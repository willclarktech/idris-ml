/* tensor_max_pool1d for the mlx backend. Strided-slice fold with
 * mx::maximum — same window-by-window pattern as avg_pool. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_max_pool1d_mlx_streamed(TensorHandle hinput, int kL, int stride,
                                                       int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	int const C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
	int const oL = (L - kL) / stride + 1;

	mx::array result = mx::full({C, oL}, -1e30, inp->data.dtype());
	for (int kl = 0; kl < kL; kl++) {
		auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
		result = mx::maximum(result, sliced);
	}

	auto* r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) {
		int const idx = tape_append(OP_MAX_POOL1D, r, inp, nullptr, 0);
		if (idx >= 0) {
			auto* meta = new MaxPool1DReplayMeta();
			meta->C = C;
			meta->L = L;
			meta->kL = kL;
			meta->stride = stride;
			meta->oL = oL;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
	return tensor_max_pool1d_mlx_streamed(hinput, kL, stride, default_stream_tag());
}

static void mlx_replay_max_pool1d(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* pm = (MaxPool1DReplayMeta*)e.meta;
	mx::array res = mx::full({pm->C, pm->oL}, -1e30, a.dtype());
	for (int kl = 0; kl < pm->kL; kl++) {
		auto sliced = mx::slice(a, {0, kl}, {pm->C, kl + pm->oL * pm->stride}, {1, pm->stride});
		res = mx::maximum(res, sliced);
	}
	pool[out] = res;
}
MLX_REGISTER_REPLAY(OP_MAX_POOL1D, mlx_replay_max_pool1d)

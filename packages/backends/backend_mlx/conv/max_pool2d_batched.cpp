/* tensor_max_pool2d_batched for the mlx backend.
 *
 * Same strided-slice fold as the per-sample variant, but with a
 * leading batch dim that's preserved by widening the slice spec. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_max_pool2d_batched_mlx_streamed(TensorHandle hinput, int kH, int kW,
                                                               int strideH, int strideW,
                                                               int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	int B = (int)inp->data.shape(0), C = (int)inp->data.shape(1);
	int H = (int)inp->data.shape(2), W = (int)inp->data.shape(3);
	int const oH = (H - kH) / strideH + 1;
	int const oW = (W - kW) / strideW + 1;

	mx::array result = mx::full({B, C, oH, oW}, -1e30, inp->data.dtype());
	for (int kh = 0; kh < kH; kh++) {
		for (int kw = 0; kw < kW; kw++) {
			auto sliced =
			    mx::slice(inp->data, {0, 0, kh, kw}, {B, C, kh + oH * strideH, kw + oW * strideW},
			              {1, 1, strideH, strideW});
			result = mx::maximum(result, sliced);
		}
	}

	auto* r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) {
		int const idx = tape_append(OP_MAX_POOL2D_BATCHED, r, inp, nullptr, 0);
		if (idx >= 0) {
			auto* meta = new MaxPool2DBatchedReplayMeta();
			meta->B = B;
			meta->C = C;
			meta->H = H;
			meta->W = W;
			meta->kH = kH;
			meta->kW = kW;
			meta->strH = strideH;
			meta->strW = strideW;
			meta->oH = oH;
			meta->oW = oW;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW, int strideH,
                                                  int strideW) {
	return tensor_max_pool2d_batched_mlx_streamed(hinput, kH, kW, strideH, strideW,
	                                              default_stream_tag());
}

static void mlx_replay_max_pool2d_batched(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* pm = (MaxPool2DBatchedReplayMeta*)e.meta;
	mx::array res = mx::full({pm->B, pm->C, pm->oH, pm->oW}, -1e30, a.dtype());
	for (int kh = 0; kh < pm->kH; kh++) {
		for (int kw = 0; kw < pm->kW; kw++) {
			auto sliced = mx::slice(a, {0, 0, kh, kw},
			                        {pm->B, pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
			                        {1, 1, pm->strH, pm->strW});
			res = mx::maximum(res, sliced);
		}
	}
	pool[out] = res;
}
MLX_REGISTER_REPLAY(OP_MAX_POOL2D_BATCHED, mlx_replay_max_pool2d_batched)

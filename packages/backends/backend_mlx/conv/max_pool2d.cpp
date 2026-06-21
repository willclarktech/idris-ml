/* tensor_max_pool2d for the mlx backend (per-sample variant). */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_max_pool2d_mlx_streamed(TensorHandle hinput, int kH, int kW,
                                                       int strideH, int strideW, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	int const C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
	int const oH = (H - kH) / strideH + 1;
	int const oW = (W - kW) / strideW + 1;

	mx::array result = mx::full({C, oH, oW}, -1e30, inp->data.dtype());
	for (int kh = 0; kh < kH; kh++) {
		for (int kw = 0; kw < kW; kw++) {
			auto sliced =
			    mx::slice(inp->data, {0, kh, kw}, {C, kh + oH * strideH, kw + oW * strideW},
			              {1, strideH, strideW});
			result = mx::maximum(result, sliced);
		}
	}

	auto* r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) {
		int const idx = tape_append(OP_MAX_POOL2D, r, inp, nullptr, 0);
		if (idx >= 0) {
			auto* meta = new MaxPool2DReplayMeta();
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

extern "C" TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW, int strideH,
                                          int strideW) {
	return tensor_max_pool2d_mlx_streamed(hinput, kH, kW, strideH, strideW, default_stream_tag());
}

static void mlx_replay_max_pool2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* pm = (MaxPool2DReplayMeta*)e.meta;
	mx::array res = mx::full({pm->C, pm->oH, pm->oW}, -1e30, a.dtype());
	for (int kh = 0; kh < pm->kH; kh++) {
		for (int kw = 0; kw < pm->kW; kw++) {
			auto sliced =
			    mx::slice(a, {0, kh, kw}, {pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
			              {1, pm->strH, pm->strW});
			res = mx::maximum(res, sliced);
		}
	}
	pool[out] = res;
}
MLX_REGISTER_REPLAY(OP_MAX_POOL2D, mlx_replay_max_pool2d)

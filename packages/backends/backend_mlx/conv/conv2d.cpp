/* tensor_conv2d (per-sample) and tensor_conv2d_batched (batch-native)
 * for the mlx backend.
 *
 * mlx::conv2d expects NHWC layout (input [N, H, W, C_in], weight
 * [C_out, kH, kW, C_in]); the Idris-side ABI uses torch's NCHW so we
 * transpose on both ends. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_conv2d_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                                                   TensorHandle hbias, int padH, int padW,
                                                   int strideH, int strideW, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	auto* ker = (Tensor*)hkernel;
	Tensor const* bias = (hbias != nullptr) ? (Tensor*)hbias : nullptr;

	int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);

	auto inp_hwc = mx::transpose(inp->data, {1, 2, 0});
	auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC});
	auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});

	auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW});
	auto out_sq = mx::squeeze(out, 0);
	auto result = mx::transpose(out_sq, {2, 0, 1});
	if (bias != nullptr) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));

	bool const rg =
	    inp->requires_grad || ker->requires_grad || ((bias != nullptr) && bias->requires_grad);
	auto* r = new Tensor(result, rg);
	if (rg) {
		int const idx = tape_append(OP_CONV2D, r, inp, ker, 0);
		if (idx >= 0) {
			auto* meta = new Conv2DReplayMeta();
			meta->padH = padH;
			meta->padW = padW;
			meta->strH = strideH;
			meta->strW = strideW;
			meta->inC = inC;
			meta->H = H;
			meta->W = W;
			meta->bias_pool_idx = (bias != nullptr) ? bias->pool_idx : -1;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                      int padH, int padW, int strideH, int strideW) {
	return tensor_conv2d_mlx_streamed(hinput, hkernel, hbias, padH, padW, strideH, strideW,
	                                  default_stream_tag());
}

extern "C" TensorHandle tensor_conv2d_batched_mlx_streamed(TensorHandle hinput,
                                                           TensorHandle hkernel, TensorHandle hbias,
                                                           int padH, int padW, int strideH,
                                                           int strideW, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	auto* ker = (Tensor*)hkernel;
	Tensor const* bias = (hbias != nullptr) ? (Tensor*)hbias : nullptr;

	int B = (int)inp->data.shape(0), inC = (int)inp->data.shape(1);
	int H = (int)inp->data.shape(2), W = (int)inp->data.shape(3);

	auto inp_nhwc = mx::transpose(inp->data, {0, 2, 3, 1});
	auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});

	auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW});
	auto result = mx::transpose(out, {0, 3, 1, 2});
	if (bias != nullptr) result = mx::add(result, mx::reshape(bias->data, {1, -1, 1, 1}));

	bool const rg =
	    inp->requires_grad || ker->requires_grad || ((bias != nullptr) && bias->requires_grad);
	auto* r = new Tensor(result, rg);
	if (rg) {
		int const idx = tape_append(OP_CONV2D_BATCHED, r, inp, ker, 0);
		if (idx >= 0) {
			auto* meta = new Conv2DBatchedReplayMeta();
			meta->padH = padH;
			meta->padW = padW;
			meta->strH = strideH;
			meta->strW = strideW;
			meta->B = B;
			meta->inC = inC;
			meta->H = H;
			meta->W = W;
			meta->bias_pool_idx = (bias != nullptr) ? bias->pool_idx : -1;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int padH, int padW, int strideH,
                                              int strideW) {
	return tensor_conv2d_batched_mlx_streamed(hinput, hkernel, hbias, padH, padW, strideH, strideW,
	                                          default_stream_tag());
}

static void mlx_replay_conv2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* cm = (Conv2DReplayMeta*)e.meta;
	int inC = cm->inC, HH = cm->H, WW = cm->W;
	auto inp_hwc = mx::transpose(a, {1, 2, 0});
	auto inp_nhwc = mx::reshape(inp_hwc, {1, HH, WW, inC});
	auto ker_mlx = mx::transpose(b, {0, 2, 3, 1});
	auto cv = mx::conv2d(inp_nhwc, ker_mlx, {cm->strH, cm->strW}, {cm->padH, cm->padW});
	auto cv_sq = mx::squeeze(cv, 0);
	auto cv_out = mx::transpose(cv_sq, {2, 0, 1});
	if (cm->bias_pool_idx >= 0) {
		cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1, 1}));
	}
	pool[out] = cv_out;
}
MLX_REGISTER_REPLAY(OP_CONV2D, mlx_replay_conv2d)

static void mlx_replay_conv2d_batched(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* cm = (Conv2DBatchedReplayMeta*)e.meta;
	int B = cm->B, inC = cm->inC, HH = cm->H, WW = cm->W;
	(void)inC;
	(void)HH;
	(void)WW; // dimensions inferred from shape
	auto inp_nhwc = mx::transpose(a, {0, 2, 3, 1});
	auto ker_mlx = mx::transpose(b, {0, 2, 3, 1});
	auto cv = mx::conv2d(inp_nhwc, ker_mlx, {cm->strH, cm->strW}, {cm->padH, cm->padW});
	auto cv_out = mx::transpose(cv, {0, 3, 1, 2});
	if (cm->bias_pool_idx >= 0) {
		cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {1, -1, 1, 1}));
	}
	(void)B;
	pool[out] = cv_out;
}
MLX_REGISTER_REPLAY(OP_CONV2D_BATCHED, mlx_replay_conv2d_batched)

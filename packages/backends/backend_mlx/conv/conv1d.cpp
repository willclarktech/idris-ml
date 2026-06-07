/* tensor_conv1d for the mlx backend.
 *
 * mlx::conv1d expects NLC layout (input [N, L, C_in], weight
 * [C_out, kL, C_in]); transpose + reshape on the way in, then
 * transpose back on the way out. The Idris-side ABI is the per-sample
 * [inC, L] / [outC, inC, kL] layout that matches torch. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_conv1d_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                                                   TensorHandle hbias, int pad, int stride,
                                                   int stream_tag) {
	WITH_STREAM(stream_tag);
	auto inp = (Tensor*)hinput;
	auto ker = (Tensor*)hkernel;
	Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
	int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);

	auto inp_lc = mx::transpose(inp->data, {1, 0}); /* [L, inC]      */
	auto inp_nlc = mx::reshape(inp_lc, {1, L, inC});
	auto ker_mlx = mx::transpose(ker->data, {0, 2, 1}); /* [outC, kL, inC] */
	auto out = mx::conv1d(inp_nlc, ker_mlx, stride, pad);
	auto out_sq = mx::squeeze(out, 0);           /* [oL, outC]     */
	auto result = mx::transpose(out_sq, {1, 0}); /* [outC, oL]     */
	if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1}));

	bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
	auto r = new Tensor(result, rg);
	if (rg) {
		int idx = tape_append(OP_CONV1D, r, inp, ker, 0);
		if (idx >= 0) {
			auto* meta = new Conv1DReplayMeta();
			meta->pad = pad;
			meta->stride = stride;
			meta->inC = inC;
			meta->L = L;
			meta->bias_pool_idx = bias ? bias->pool_idx : -1;
			tape[idx].meta = meta;
		}
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                      int pad, int stride) {
	return tensor_conv1d_mlx_streamed(hinput, hkernel, hbias, pad, stride, default_stream_tag());
}

static void mlx_replay_conv1d(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* cm = (Conv1DReplayMeta*)e.meta;
	int inC = cm->inC, LL = cm->L;
	auto inp_lc = mx::transpose(a, {1, 0});
	auto inp_nlc = mx::reshape(inp_lc, {1, LL, inC});
	auto ker_mlx = mx::transpose(b, {0, 2, 1});
	auto cv = mx::conv1d(inp_nlc, ker_mlx, cm->stride, cm->pad);
	auto cv_sq = mx::squeeze(cv, 0);
	auto cv_out = mx::transpose(cv_sq, {1, 0});
	if (cm->bias_pool_idx >= 0)
		cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1}));
	pool[out] = cv_out;
}
MLX_REGISTER_REPLAY(OP_CONV1D, mlx_replay_conv1d)

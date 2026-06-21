/* tensor_avg_pool1d for the mlx backend.
 *
 * No native avg_pool1d in mlx — built from a strided-slice fold: for
 * each kernel offset kl, slice the input with that offset + stride
 * and accumulate, then divide by kL. */
#include <cmath>
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_avg_pool1d_mlx_streamed(TensorHandle hinput, int kL, int stride,
                                                       int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* inp = (Tensor*)hinput;
	auto dt = inp->data.dtype();
	int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
	int const oL = (L - kL) / stride + 1;
	mx::array result = mx::zeros({C, oL}, dt);
	for (int kl = 0; kl < kL; kl++) {
		auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
		result = mx::add(result, sliced);
	}
	result = mx::divide(result, mx::array((double)kL, dt));
	auto* r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad)
		tape_append(OP_AVG_POOL1D, r, inp, nullptr, (double)kL + stride * 0.001);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
	return tensor_avg_pool1d_mlx_streamed(hinput, kL, stride, default_stream_tag());
}

static void mlx_replay_avg_pool1d(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	// scalar_arg encodes kL + stride*0.001
	int const kL = (int)e.scalar_arg;
	int stride = (int)std::lround((e.scalar_arg - kL) * 1000);
	if (stride == 0) stride = kL;
	int const oL = ((int)a.shape(1) - kL) / stride + 1;
	mx::array res = mx::zeros({(int)a.shape(0), oL}, a.dtype());
	for (int kl = 0; kl < kL; kl++) {
		auto sliced = mx::slice(a, {0, kl}, {(int)a.shape(0), kl + oL * stride}, {1, stride});
		res = mx::add(res, sliced);
	}
	pool[out] = mx::divide(res, scalar_like((double)kL, a));
}
MLX_REGISTER_REPLAY(OP_AVG_POOL1D, mlx_replay_avg_pool1d)

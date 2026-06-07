/* tensor_conv1d_circular for the mlx backend.
 *
 * Circular convolution: out[i] = sum_j input[(i - k/2 + j + n) % n] * kernel[k-1-j].
 * Note the reversed kernel index — this is convolution, not correlation, matching
 * tape's reference at backend_tape/conv/conv1d_circular.c.
 * mlx has no native circular conv1d — built from per-shift roll + multiply +
 * accumulate. OP_CONV1D_CIRC replay reproduces the rolls. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../training/autograd/op_dispatch.h"
#include "../precision.h"

extern "C" TensorHandle tensor_conv1d_circular_mlx_streamed(TensorHandle hinput,
                                                            TensorHandle hkernel, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto inp = (Tensor*)hinput;
	auto kern = (Tensor*)hkernel;
	int n = (int)inp->data.size();
	int k = (int)kern->data.size();

	mx::array result = mx::zeros({n}, inp->data.dtype());
	int half_k = k / 2;
	for (int j = 0; j < k; j++) {
		int shift = half_k - j;
		auto shifted = mx::roll(inp->data, shift);
		auto kern_j = mx::take(kern->data, mx::array(k - 1 - j));
		result = mx::add(result, mx::multiply(shifted, kern_j));
	}

	bool rg = inp->requires_grad || kern->requires_grad;
	auto r = new Tensor(result, rg);
	if (rg) tape_append(OP_CONV1D_CIRC, r, inp, kern, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
	return tensor_conv1d_circular_mlx_streamed(hinput, hkernel, default_stream_tag());
}

static void mlx_replay_conv1d_circ(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	// Inline circular convolution forward (kernel reversed — see
	// backend_mlx/conv/conv1d_circular.cpp for the index derivation)
	int n = (int)a.size(), k = (int)b.size();
	int half_k = k / 2;
	auto result = mx::zeros({n}, a.dtype());
	for (int j = 0; j < k; j++) {
		auto shifted = mx::roll(a, half_k - j);
		auto kern_j = mx::take(b, mx::array(k - 1 - j));
		result = mx::add(result, mx::multiply(shifted, kern_j));
	}
	pool[out] = result;
}
MLX_REGISTER_REPLAY(OP_CONV1D_CIRC, mlx_replay_conv1d_circ)

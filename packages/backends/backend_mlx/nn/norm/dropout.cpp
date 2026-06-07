/* tensor_dropout for the mlx backend.
 *
 * MLX random only supports float32 on Metal — generate the bernoulli
 * mask in f32, compare, scale, then cast to the input's dtype so the
 * final multiply doesn't force a dtype promotion. The mask is stored
 * as a non-grad arg2 so vjp can differentiate through the multiply. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h" /* kF32_ZERO / kF32_ONE */
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_dropout_mlx_streamed(TensorHandle hinput, double p, int training,
                                                    unsigned int seed, int stream_tag) {
	WITH_STREAM(stream_tag);
	(void)seed;
	auto inp = (Tensor*)hinput;
	if (!training || p <= 0.0) return hinput;

	double scale = 1.0 / (1.0 - p);
	auto rnd = mx::random::uniform(kF32_ZERO(), kF32_ONE(), inp->data.shape(), mx::float32);
	auto keep = mx::greater(rnd, mx::array((float)p, mx::float32));
	auto mask_f32 = mx::where(keep, mx::array(scale, mx::float32), kF32_ZERO());
	auto mask = mx::astype(mask_f32, inp->data.dtype());
	auto result = mx::multiply(inp->data, mask);

	auto r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) {
		auto mask_t = new Tensor(mask, false);
		tape_append(OP_DROPOUT, r, inp, mask_t, 0);
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_dropout(TensorHandle hinput, double p, int training,
                                       unsigned int seed) {
	return tensor_dropout_mlx_streamed(hinput, p, training, seed, default_stream_tag());
}

static void mlx_replay_dropout(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	// b holds the stored mask tensor; just multiply
	pool[out] = mx::multiply(a, b);
}
MLX_REGISTER_REPLAY(OP_DROPOUT, mlx_replay_dropout)

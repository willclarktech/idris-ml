/* tensor_log_softmax + 2d variant for the mlx backend.
 *
 * mlx has no native log_softmax — implement via the LSE identity
 * (log_softmax(x) = x - log(sum(exp(x)))) which is numerically stable
 * when expressed via the max-shift form. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_log_softmax_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto maxv = mx::max(t->data, dim, true);
	auto shifted = mx::subtract(t->data, maxv);
	auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
	auto r = new Tensor(mx::subtract(t->data, lse), t->requires_grad);
	if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, (double)dim);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
	return tensor_log_softmax_mlx_streamed(h, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_log_softmax_2d_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto maxv = mx::max(t->data, -1, true);
	auto shifted = mx::subtract(t->data, maxv);
	auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), -1, true)), maxv);
	auto r = new Tensor(mx::subtract(t->data, lse), t->requires_grad);
	if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, -1.0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_log_softmax_2d(TensorHandle h) {
	return tensor_log_softmax_2d_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_log_softmax_2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	int dim = (int)e.scalar_arg; // stored by forward (0 for 1D, -1 for 2D)
	auto maxv = mx::max(a, dim, true);
	auto shifted = mx::subtract(a, maxv);
	auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
	pool[out] = mx::subtract(a, lse);
}
MLX_REGISTER_REPLAY(OP_LOG_SOFTMAX_2D, mlx_replay_log_softmax_2d)

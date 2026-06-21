/* tensor_transpose_2d / tensor_transpose_last2 for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_transpose_2d_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* t = (Tensor*)h;
	auto* r = new Tensor(mx::transpose(t->data, {1, 0}), t->requires_grad);
	if (t->requires_grad) tape_append(OP_TRANSPOSE_2D, r, t, nullptr, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_transpose_2d(TensorHandle h) {
	return tensor_transpose_2d_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_transpose_last2_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* t = (Tensor*)h;
	auto* r = new Tensor(mx::transpose(t->data, {0, 2, 1}), t->requires_grad);
	if (t->requires_grad) tape_append(OP_TRANSPOSE_LAST2, r, t, nullptr, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_transpose_last2(TensorHandle h) {
	return tensor_transpose_last2_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_transpose_last2(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::transpose(a, {0, 2, 1});
}
MLX_REGISTER_REPLAY(OP_TRANSPOSE_LAST2, mlx_replay_transpose_last2)

static void mlx_replay_transpose_2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::transpose(a, {1, 0});
}
MLX_REGISTER_REPLAY(OP_TRANSPOSE_2D, mlx_replay_transpose_2d)

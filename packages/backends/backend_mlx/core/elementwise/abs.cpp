/* tensor_abs for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_abs_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* t = (Tensor*)h;
	auto* r = new Tensor(mx::abs(t->data), t->requires_grad);
	if (t->requires_grad) tape_append(OP_ABS, r, t, nullptr, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_abs(TensorHandle h) {
	return tensor_abs_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_abs(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::abs(a);
}
MLX_REGISTER_REPLAY(OP_ABS, mlx_replay_abs)

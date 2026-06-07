/* tensor_log for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_log_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto r = new Tensor(mx::log(t->data), t->requires_grad);
	if (t->requires_grad) tape_append(OP_LOG, r, t, nullptr, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_log(TensorHandle h) {
	return tensor_log_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_log(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::log(a);
}
MLX_REGISTER_REPLAY(OP_LOG, mlx_replay_log)

/* tensor_outer for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_outer_mlx_streamed(TensorHandle ha, TensorHandle hb,
                                                  int stream_tag) {
	WITH_STREAM(stream_tag);
	auto a = (Tensor*)ha;
	auto b = (Tensor*)hb;
	bool rg = a->requires_grad || b->requires_grad;
	auto r = new Tensor(mx::outer(a->data, b->data), rg);
	if (rg) tape_append(OP_OUTER, r, a, b, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_outer(TensorHandle ha, TensorHandle hb) {
	return tensor_outer_mlx_streamed(ha, hb, default_stream_tag());
}

static void mlx_replay_outer(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::outer(a, b);
}
MLX_REGISTER_REPLAY(OP_OUTER, mlx_replay_outer)

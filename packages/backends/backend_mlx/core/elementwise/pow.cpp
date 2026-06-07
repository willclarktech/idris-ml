/* tensor_pow for the mlx backend (elementwise base ^ exp). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_pow_mlx_streamed(TensorHandle hbase, TensorHandle hexp,
                                                int stream_tag) {
	WITH_STREAM(stream_tag);
	auto b = (Tensor*)hbase;
	auto e = (Tensor*)hexp;
	bool rg = b->requires_grad || e->requires_grad;
	auto r = new Tensor(mx::power(b->data, e->data), rg);
	if (rg) tape_append(OP_POW, r, b, e, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_pow(TensorHandle hbase, TensorHandle hexp) {
	return tensor_pow_mlx_streamed(hbase, hexp, default_stream_tag());
}

static void mlx_replay_pow(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::power(a, b);
}
MLX_REGISTER_REPLAY(OP_POW, mlx_replay_pow)

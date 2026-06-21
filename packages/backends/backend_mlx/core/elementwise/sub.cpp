/* tensor_sub for the mlx backend. See add.cpp for the streamed + tape
 * autograd pattern shared across binary elementwise ops. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_sub_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* a = (Tensor*)ha;
	auto* b = (Tensor*)hb;
	bool const rg = a->requires_grad || b->requires_grad;
	auto* r = new Tensor(mx::subtract(a->data, b->data), rg);
	if (rg) tape_append(OP_SUB, r, a, b, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_sub(TensorHandle ha, TensorHandle hb) {
	return tensor_sub_mlx_streamed(ha, hb, default_stream_tag());
}

static void mlx_replay_sub(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::subtract(a, b);
}
MLX_REGISTER_REPLAY(OP_SUB, mlx_replay_sub)

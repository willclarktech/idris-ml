/* tensor_mul for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_mul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* a = (Tensor*)ha;
	auto* b = (Tensor*)hb;
	bool const rg = a->requires_grad || b->requires_grad;
	auto* r = new Tensor(mx::multiply(a->data, b->data), rg);
	if (rg) tape_append(OP_MUL, r, a, b, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_mul(TensorHandle ha, TensorHandle hb) {
	return tensor_mul_mlx_streamed(ha, hb, default_stream_tag());
}

static void mlx_replay_mul(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::multiply(a, b);
}
MLX_REGISTER_REPLAY(OP_MUL, mlx_replay_mul)

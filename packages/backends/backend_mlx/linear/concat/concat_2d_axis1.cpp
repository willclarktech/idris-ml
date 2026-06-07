/* tensor_concat_2d_axis1 for the mlx backend.
 * A: [m, n], B: [m, k] -> [m, n+k] along axis 1. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_concat_2d_axis1_mlx_streamed(TensorHandle hA, TensorHandle hB,
                                                            int stream_tag) {
	WITH_STREAM(stream_tag);
	auto A = (Tensor*)hA;
	auto B = (Tensor*)hB;
	bool rg = A->requires_grad || B->requires_grad;
	auto r = new Tensor(mx::concatenate({A->data, B->data}, 1), rg);
	if (rg) tape_append(OP_CONCAT_2D_AXIS1, r, A, B, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_concat_2d_axis1(TensorHandle hA, TensorHandle hB) {
	return tensor_concat_2d_axis1_mlx_streamed(hA, hB, default_stream_tag());
}

static void mlx_replay_concat_2d_axis1(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	/* a = A [m,n], b = B [m,k]. Result = concat along axis 1 -> [m,n+k] */
	pool[out] = mx::concatenate({a, b}, 1);
}
MLX_REGISTER_REPLAY(OP_CONCAT_2D_AXIS1, mlx_replay_concat_2d_axis1)

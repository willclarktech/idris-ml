/* tensor_scatter_add for the mlx backend.
 *
 * mx::scatter_add's updates shape is indices.shape + base.shape[axis+1:].
 * For a 1D base on axis 0 that's [N, 1] (the trailing 1 is the empty
 * remainder reified as a singleton) — hence the reshape. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_scatter_add_mlx_streamed(TensorHandle hindex, TensorHandle hsrc,
                                                        int out_size, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* idx = (Tensor*)hindex;
	auto* src = (Tensor*)hsrc;
	auto idx_int = mx::astype(idx->data, mx::int32);
	auto base = mx::zeros({out_size}, src->data.dtype());
	auto updates_2d = mx::reshape(src->data, {(int)src->data.size(), 1});
	auto result = mx::scatter_add(base, {idx_int}, updates_2d, std::vector<int>{0});
	auto* r = new Tensor(result, src->requires_grad);
	if (src->requires_grad) tape_append(OP_SCATTER_ADD, r, src, idx, (double)out_size);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
	return tensor_scatter_add_mlx_streamed(hindex, hsrc, out_size, default_stream_tag());
}

static void mlx_replay_scatter_add(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	int const out_size = (int)e.scalar_arg;
	// arg2 (index tensor) is always present for OP_SCATTER_ADD (tape_append
	// sets it); the analyzer's null path is infeasible.
	// NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
	auto idx_int = mx::astype(e.arg2->data, mx::int32);
	auto base = mx::zeros({out_size}, a.dtype());
	auto updates_2d = mx::reshape(a, {(int)a.size(), 1});
	pool[out] = mx::scatter_add(base, {idx_int}, updates_2d, std::vector<int>{0});
}
MLX_REGISTER_REPLAY(OP_SCATTER_ADD, mlx_replay_scatter_add)

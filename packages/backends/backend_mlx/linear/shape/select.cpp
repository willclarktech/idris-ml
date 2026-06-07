/* tensor_select for the mlx backend.
 *
 * mx::take(arr, indices_array, axis) returns a slice with that axis
 * removed when indices is a scalar — matches torch's .select() shape.
 * OP_SELECT carries the `index` in scalar_arg for backward replay. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_select_mlx_streamed(TensorHandle h, int dim, int index,
                                                   int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto r = new Tensor(mx::take(t->data, mx::array(index), dim), t->requires_grad);
	if (t->requires_grad) tape_append(OP_SELECT, r, t, nullptr, (double)index);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_select(TensorHandle h, int dim, int index) {
	return tensor_select_mlx_streamed(h, dim, index, default_stream_tag());
}

static void mlx_replay_select(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = mx::take(a, mx::array((int)e.scalar_arg), 0);
}
MLX_REGISTER_REPLAY(OP_SELECT, mlx_replay_select)

/* tensor_masked_fill for the mlx backend.
 *
 * mlx has no native masked_fill — built from mx::where on a same-shape
 * `value` constant. OP_MASKED_FILL replay reproduces the where call. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_masked_fill_mlx_streamed(TensorHandle h, TensorHandle hmask,
                                                        double value, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto mask = (Tensor*)hmask;
	auto val_arr = mx::full(t->data.shape(), value, t->data.dtype());
	auto r = new Tensor(mx::where(mask->data, val_arr, t->data), t->requires_grad);
	if (t->requires_grad) tape_append(OP_MASKED_FILL, r, t, mask, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle hmask, double value) {
	return tensor_masked_fill_mlx_streamed(h, hmask, value, default_stream_tag());
}

static void mlx_replay_masked_fill(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	/* mask `b` may be bool; the fill value must match `a`'s
	                   dtype so `mx::where` doesn't force a promotion. */
	auto kNegInfMask = scalar_like(-1e9, a);
	pool[out] = mx::where(b, kNegInfMask, a);
}
MLX_REGISTER_REPLAY(OP_MASKED_FILL, mlx_replay_masked_fill)

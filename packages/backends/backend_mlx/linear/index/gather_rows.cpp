/* tensor_gather_rows for the mlx backend. mx::take_along_axis requires
 * integer indices — coerce to int32 on the way in. OP_GATHER_ROWS
 * replay reproduces the take from the stored index tensor (arg2, which
 * `arg2_is_index` in backward.cpp excludes from the differentiable
 * inputs — indices are discrete). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_gather_rows_mlx_streamed(TensorHandle hinput, TensorHandle hindex,
                                                        int b, int n, int stream_tag) {
	WITH_STREAM(stream_tag);
	(void)b;
	(void)n;
	auto inp = (Tensor*)hinput;
	auto idx = (Tensor*)hindex;
	auto idx_int = mx::expand_dims(mx::astype(idx->data, mx::int32), 1);
	auto result = mx::squeeze(mx::take_along_axis(inp->data, idx_int, 1), 1);
	auto r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) tape_append(OP_GATHER_ROWS, r, inp, idx, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_gather_rows(TensorHandle hinput, TensorHandle hindex, int b, int n) {
	return tensor_gather_rows_mlx_streamed(hinput, hindex, b, n, default_stream_tag());
}

static void mlx_replay_gather_rows(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	// Indices are discrete and non-differentiable — read directly from
	// the tape entry's tensor (closure-captured, not via pool); the
	// constants collection excludes arg2 for this op (arg2_is_index).
	auto idx_int = mx::expand_dims(mx::astype(e.arg2->data, mx::int32), 1);
	pool[out] = mx::squeeze(mx::take_along_axis(a, idx_int, 1), 1);
}
MLX_REGISTER_REPLAY(OP_GATHER_ROWS, mlx_replay_gather_rows)

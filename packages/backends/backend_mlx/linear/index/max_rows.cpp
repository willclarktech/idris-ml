/* tensor_max_rows for the mlx backend. mx::max along axis 1; the vjp
 * routes the gradient to the argmax cells (tie-breaking unspecified
 * across backends; tests avoid ties). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_max_rows_mlx_streamed(TensorHandle hinput, int b, int n,
                                                     int stream_tag) {
	WITH_STREAM(stream_tag);
	(void)b;
	(void)n;
	auto* inp = (Tensor*)hinput;
	auto result = mx::max(inp->data, 1);
	auto* r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) tape_append(OP_MAX_ROWS, r, inp, nullptr, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_max_rows(TensorHandle hinput, int b, int n) {
	return tensor_max_rows_mlx_streamed(hinput, b, n, default_stream_tag());
}

static void mlx_replay_max_rows(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	pool[out] = mx::max(a, 1);
}
MLX_REGISTER_REPLAY(OP_MAX_ROWS, mlx_replay_max_rows)

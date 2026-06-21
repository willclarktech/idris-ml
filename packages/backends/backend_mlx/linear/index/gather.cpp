/* tensor_gather for the mlx backend. mx::take requires int32 indices —
 * coerce on the way in. OP_GATHER replay reproduces the take from the
 * stored index tensor (arg2). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_gather_mlx_streamed(TensorHandle hinput, TensorHandle hindex, int n,
                                                   int stream_tag) {
	WITH_STREAM(stream_tag);
	(void)n;
	auto* inp = (Tensor*)hinput;
	auto* idx = (Tensor*)hindex;
	auto idx_int = mx::astype(idx->data, mx::int32);
	auto result = mx::take(inp->data, idx_int, 0);
	auto* r = new Tensor(result, inp->requires_grad);
	if (inp->requires_grad) tape_append(OP_GATHER, r, inp, idx, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
	return tensor_gather_mlx_streamed(hinput, hindex, n, default_stream_tag());
}

static void mlx_replay_gather(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	// Indices are discrete and non-differentiable — read directly
	// from the tape entry's tensor (closure-captured, not via
	// pool). The constants-collection above intentionally
	// excludes arg2 for this op so mlx::vjp never sees it as a
	// differentiable input. See `arg2_is_index` above.
	// arg2 (index tensor) is always present for OP_GATHER (tape_append sets
	// it); the analyzer's null path is infeasible.
	// NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
	auto idx_int = mx::astype(e.arg2->data, mx::int32);
	pool[out] = mx::take(a, idx_int, 0);
}
MLX_REGISTER_REPLAY(OP_GATHER, mlx_replay_gather)

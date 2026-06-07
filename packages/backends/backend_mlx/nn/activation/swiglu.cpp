/* tensor_swiglu_2d for the mlx backend.
 *
 *   out = silu(gate) * up = gate * sigmoid(gate) * up
 *
 * mlx has no native fused swiglu; composing the three primitives into a
 * single tape entry still collapses the per-call wrap/append overhead
 * and gives mlx's compiler one expression-tree to fuse.
 *
 * Replaces the tsilu + tmul pair in HfLlama.applyMlp.
 */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

static mx::array swiglu_compute(const mx::array& g, const mx::array& u) {
	return mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
}

extern "C" TensorHandle tensor_swiglu_2d_mlx_streamed(TensorHandle hgate, TensorHandle hup,
                                                      int stream_tag) {
	WITH_STREAM(stream_tag);
	auto g = (Tensor*)hgate;
	auto u = (Tensor*)hup;
	auto result = swiglu_compute(g->data, u->data);
	bool rg = g->requires_grad || u->requires_grad;
	auto r = new Tensor(result, rg);
	if (rg) tape_append(OP_SWIGLU_2D, r, g, u, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_swiglu_2d(TensorHandle hgate, TensorHandle hup) {
	return tensor_swiglu_2d_mlx_streamed(hgate, hup, default_stream_tag());
}

static void mlx_replay_swiglu_2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	auto g = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	auto u = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	pool[out] = swiglu_compute(g, u);
}
MLX_REGISTER_REPLAY(OP_SWIGLU_2D, mlx_replay_swiglu_2d)

/* tensor_softmax_xent_2d for the mlx backend.
 *
 * Fused softmax cross-entropy with logits (soft/one-hot targets):
 *   out = -scale * sum(target * log_softmax(input, rows))
 * log_softmax via the max-shifted LSE identity (mlx has no native
 * log_softmax; same form as nn/softmax/log_softmax.cpp) — smooth
 * everywhere, so the replay-based vjp needs no closed-form special
 * case (contrast bce_with_logits' kink note). ONE tape entry; the
 * scale rides scalar_arg, target rides arg2 — no ReplayMeta needed. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

static mx::array softmax_xent_expr(const mx::array& in, const mx::array& tg, double scale) {
	auto maxv = mx::max(in, -1, true);
	auto shifted = mx::subtract(in, maxv);
	auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), -1, true)), maxv);
	auto ls = mx::subtract(in, lse);
	auto acc = mx::sum(mx::multiply(tg, ls));
	return mx::multiply(mx::negative(acc), scalar_like(scale, in));
}

extern "C" TensorHandle tensor_softmax_xent_2d_mlx_streamed(TensorHandle hinput,
                                                            TensorHandle htarget, double scale,
                                                            int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* in = (Tensor*)hinput;
	auto* tgt = (Tensor*)htarget;
	auto result = softmax_xent_expr(in->data, tgt->data, scale);
	bool const rg = in->requires_grad || tgt->requires_grad;
	auto* r = new Tensor(result, rg);
	if (rg) tape_append(OP_SOFTMAX_XENT_2D, r, in, tgt, scale);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_softmax_xent_2d(TensorHandle hinput, TensorHandle htarget,
                                               double scale) {
	return tensor_softmax_xent_2d_mlx_streamed(hinput, htarget, scale, default_stream_tag());
}

static void mlx_replay_softmax_xent_2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO(); // logits
	auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO(); // targets
	pool[out] = softmax_xent_expr(a, b, e.scalar_arg);
}
MLX_REGISTER_REPLAY(OP_SOFTMAX_XENT_2D, mlx_replay_softmax_xent_2d)

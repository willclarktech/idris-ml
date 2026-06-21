/* tensor_bce_with_logits for the mlx backend.
 *
 * BCE with logits = mean(max(p,0) - p*y + log(1 + exp(-|p|))) — the
 * numerically-stable composite (forward value correct for all p). It is
 * recorded as ONE tape entry (OP_BCE_WITH_LOGITS) so the replay-based
 * vjp produces the exact closed-form gradient (sigmoid(p) - y)/n.
 *
 * Why not the old decomposed-primitive path: max(p,0) and |p| are
 * non-differentiable at p=0, and mlx's vjp picks subgradient 0 there, so
 * the decomposed replay gave d/dp = -1.0 at the kink instead of
 * sigmoid(0)-y = -0.5. Tape (closed form) and torch (libtorch fused)
 * agree at the kink; this aligns mlx with them. Identical fix to
 * OP_SOFTPLUS — see core/elementwise/softplus.cpp. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_bce_with_logits_mlx_streamed(TensorHandle hinput,
                                                            TensorHandle htarget, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* in = (Tensor*)hinput;
	auto* tgt = (Tensor*)htarget;
	auto p = in->data;
	auto y = tgt->data;
	// Stable composite: max(p,0) - p*y + log(1 + exp(-|p|)), then mean.
	// Raw mx:: ops record no sub-tape-entries — the single tape_append
	// below owns the backward.
	auto inner = mx::add(mx::subtract(mx::maximum(p, zero_like(p)), mx::multiply(p, y)),
	                     mx::log(mx::add(one_like(p), mx::exp(mx::negative(mx::abs(p))))));
	auto result = mx::mean(inner);
	bool const rg = in->requires_grad || tgt->requires_grad;
	auto* r = new Tensor(result, rg);
	if (rg) tape_append(OP_BCE_WITH_LOGITS, r, in, tgt, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_bce_with_logits(TensorHandle hinput, TensorHandle htarget) {
	return tensor_bce_with_logits_mlx_streamed(hinput, htarget, default_stream_tag());
}

static void mlx_replay_bce_with_logits(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO(); // logits p
	auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO(); // target y
	// Smooth form mean(softplus(p) - p*y) = mean(log(1+exp(p)) - p*y),
	// since max(p,0) + log(1+exp(-|p|)) ≡ log(1+exp(p)). Differentiable
	// everywhere; mx::vjp gives d/dp = (sigmoid(p) - y)/n, correct at the
	// p=0 kink. The stable max/abs composite can't be reused here — its
	// vjp returns subgradient 0 at p=0 (same trap as OP_SOFTPLUS).
	pool[out] =
	    mx::mean(mx::subtract(mx::log(mx::add(one_like(a), mx::exp(a))), mx::multiply(a, b)));
}
MLX_REGISTER_REPLAY(OP_BCE_WITH_LOGITS, mlx_replay_bce_with_logits)

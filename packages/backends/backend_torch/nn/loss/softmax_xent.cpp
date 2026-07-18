/* tensor_softmax_xent_2d for the torch backend.
 *
 * Fused softmax cross-entropy with logits (soft/one-hot targets):
 *   out = -scale * sum(target * log_softmax(input, rows))
 * Composed from at:: ops — libtorch autograd tracks each step, so no
 * hand-written backward. dim=-1 is row-wise for [b, n] and the whole
 * vector for the rank-1 [n] case, matching tape's convention.
 * Replaces the decomposed log_softmax -> mul -> sum -> neg ->
 * mul_scalar chain (and the forward-only tensor_cross_entropy this
 * file's sibling kept as a placeholder). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_softmax_xent_2d(TensorHandle input, TensorHandle target,
                                               double scale) {
	auto& in = *to_tensor(input);
	auto& tg = *to_tensor(target);
	auto ls = at::log_softmax(in, -1);
	auto loss = (tg * ls).sum() * (-scale);
	return from_tensor(loss);
}

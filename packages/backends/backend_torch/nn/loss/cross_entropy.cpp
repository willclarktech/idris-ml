/* tensor_cross_entropy for the torch backend.
 *
 * Match tape's convention: -sum(target * log_softmax(input)) / numel.
 * This differs from torch::nn::functional::cross_entropy (which expects
 * targets as class indices and means over the batch dim). The Idris
 * side has no caller for this symbol today — the future fused
 * softmax_cross_entropy_with_logits will replace it; this impl just
 * keeps the two backends agreeing for the regression suite. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_cross_entropy(TensorHandle input, TensorHandle target) {
	auto& in = *to_tensor(input);
	auto& tg = *to_tensor(target);
	auto ls = at::log_softmax(in, 0);
	auto loss = -(tg * ls).sum() / static_cast<double>(ls.numel());
	return from_tensor(loss);
}

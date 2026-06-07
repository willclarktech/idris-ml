/* tensor_cross_attention for the torch backend.
 *
 * Standard scaled-dot-product attention composed from bmm + masked_fill
 * + softmax + bmm. The Idris-side smart constructor passes an unscaled
 * scale (1/sqrt(d_k)) so this body just multiplies. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                               TensorHandle hmask, double scale) {
	auto& Q = *to_tensor(hQ);
	auto& K = *to_tensor(hK);
	auto& V = *to_tensor(hV);
	auto scores = torch::bmm(Q, K.transpose(-2, -1)) * scale;
	if (hmask) scores = scores.masked_fill(to_tensor(hmask)->to(torch::kBool), -1.0e20);
	auto attn = torch::softmax(scores, -1);
	return from_tensor(torch::bmm(attn, V));
}

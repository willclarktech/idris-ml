/* tensor_expand_mask for the mlx backend. Mask broadcasts don't carry
 * grad — the mask is always a constant. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_expand_mask_mlx_streamed(TensorHandle hmask, int B, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* mask = (Tensor*)hmask;
	int const m = mask->data.shape(0), n = mask->data.shape(1);
	/* [m,n] → [1,m,n] → broadcast to [B,m,n] */
	auto expanded = mx::broadcast_to(mx::reshape(mask->data, {1, m, n}), {B, m, n});
	auto* r = new Tensor(expanded, false);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
	return tensor_expand_mask_mlx_streamed(hmask, B, default_stream_tag());
}

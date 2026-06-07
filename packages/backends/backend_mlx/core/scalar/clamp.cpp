/* tensor_clamp for the mlx backend. Two-sided scalar clamp via
 * mx::clip. Inference-only; no tape entry / no replay. */
#include "../../tensor.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_clamp_mlx_streamed(TensorHandle h, double lo, double hi,
                                                  int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto r = new Tensor(mx::clip(t->data, scalar_like(lo, t->data), scalar_like(hi, t->data)),
	                    /*requires_grad=*/false);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_clamp(TensorHandle h, double lo, double hi) {
	return tensor_clamp_mlx_streamed(h, lo, hi, default_stream_tag());
}

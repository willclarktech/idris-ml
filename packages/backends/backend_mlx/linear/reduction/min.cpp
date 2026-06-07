/* tensor_min for the mlx backend. Non-grad — min selects an entry, not
 * a smooth function of the input — so we don't append a tape entry. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_min_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto result = mx::min(t->data);
	mx::eval(result);
	return (TensorHandle) new Tensor(result, false);
}

extern "C" TensorHandle tensor_min(TensorHandle h) {
	return tensor_min_mlx_streamed(h, default_stream_tag());
}

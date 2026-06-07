/* tensor_max for the mlx backend. Non-grad — see min.cpp. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_max_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto result = mx::max(t->data);
	mx::eval(result);
	return (TensorHandle) new Tensor(result, false);
}

extern "C" TensorHandle tensor_max(TensorHandle h) {
	return tensor_max_mlx_streamed(h, default_stream_tag());
}

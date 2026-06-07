/* tensor_clone for the mlx backend. Constructs a new Tensor wrapping
 * a fresh mx::array referencing the same data — mx::array is COW, so
 * any subsequent mutation goes through copy-on-write semantics. The
 * result is non-tracking (requires_grad=false) like tape's clone. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_clone_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto c = new Tensor(mx::array(t->data), false);
	return (TensorHandle)c;
}

extern "C" TensorHandle tensor_clone(TensorHandle h) {
	return tensor_clone_mlx_streamed(h, default_stream_tag());
}

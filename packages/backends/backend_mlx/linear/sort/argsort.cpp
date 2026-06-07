/* tensor_argsort for the mlx backend. mx::argsort returns ascending
 * indices; reverse manually for descending=1. Result is cast to F32
 * (mlx's default index dtype on the autograd surface) and grad-free. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_argsort_mlx_streamed(TensorHandle ht, int dim, int descending,
                                                    int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)ht;
	auto indices = mx::argsort(t->data, dim);
	if (descending) {
		int n = (int)t->data.size();
		auto rev_idx = mx::subtract(mx::array(n - 1), mx::arange(n));
		indices = mx::take(indices, rev_idx);
	}
	auto result = mx::astype(indices, mx::float32);
	mx::eval(result);
	return (TensorHandle)(new Tensor(result, false));
}

extern "C" TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
	return tensor_argsort_mlx_streamed(ht, dim, descending, default_stream_tag());
}

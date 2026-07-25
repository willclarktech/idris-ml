/* tensor_argsort for the mlx backend. mx::argsort returns ascending
 * indices; descending sorts the negated values (stable, ties ascending
 * by index — see the tape comparators). Result is cast to F32
 * (mlx's default index dtype on the autograd surface) and grad-free. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_argsort_mlx_streamed(TensorHandle ht, int dim, int descending,
                                                    int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* t = (Tensor*)ht;
	// Descending sorts the negated values instead of reversing the
	// ascending indices: mlx's sort is stable, and reversal would put
	// tied runs in descending index order while the tape/torch backends
	// tie-break ascending (the DNC allocation weighting depends on it).
	auto indices =
	    (descending != 0) ? mx::argsort(mx::negative(t->data), dim) : mx::argsort(t->data, dim);
	auto result = mx::astype(indices, mx::float32);
	mx::eval(result);
	return (TensorHandle)(new Tensor(result, false));
}

extern "C" TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
	return tensor_argsort_mlx_streamed(ht, dim, descending, default_stream_tag());
}

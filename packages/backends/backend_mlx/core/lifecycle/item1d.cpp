/* tensor_item_1d for the mlx backend. Flat-buffer indexing semantics
 * matching tape (tape_load_d) and torch (.flatten()[idx]): `idx` is a
 * flat offset into the data layout, not a first-dim index. */
#include "../../tensor.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" double tensor_item_1d_mlx_streamed(TensorHandle vec, int idx, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* t = (Tensor*)vec;
	/* The source may be non-contiguous (e.g. a transposed conv1d output);
	 * flatten produces a contiguous view so `idx` is a valid flat offset
	 * into the logical row-major layout, not the underlying storage order.
	 * Matches tensor_item_2d; without it a strided view reads in storage
	 * order (e.g. a multichannel conv1d output comes back [oL,outC]). */
	auto flat = mx::flatten(t->data, mx::StreamOrDevice{});
	mx::eval(flat);
	return mx_read_double(flat, idx);
}

extern "C" double tensor_item_1d(TensorHandle vec, int idx) {
	return tensor_item_1d_mlx_streamed(vec, idx, default_stream_tag());
}

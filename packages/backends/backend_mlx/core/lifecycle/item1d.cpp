/* tensor_item_1d for the mlx backend. Flat-buffer indexing semantics
 * matching tape (tape_load_d) and torch (.flatten()[idx]): `idx` is a
 * flat offset into the data layout, not a first-dim index. */
#include "../../tensor.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" double tensor_item_1d_mlx_streamed(TensorHandle vec, int idx, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)vec;
    mx::eval(t->data);
    return mx_read_double(t->data, idx);
}

extern "C" double tensor_item_1d(TensorHandle vec, int idx) {
    return tensor_item_1d_mlx_streamed(vec, idx, default_stream_tag());
}

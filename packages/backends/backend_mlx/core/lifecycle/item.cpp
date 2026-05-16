/* tensor_item for the mlx backend. mx::eval forces lazy evaluation
 * before the host readback; the dtype branch widens f32 to double. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" double tensor_item_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    mx::eval(t->data);
    if (t->data.dtype() == mx::float64) return t->data.item<double>();
    return (double)t->data.item<float>();
}

extern "C" double tensor_item(TensorHandle h) {
    return tensor_item_mlx_streamed(h, default_stream_tag());
}

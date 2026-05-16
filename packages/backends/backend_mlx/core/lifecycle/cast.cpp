/* tensor_cast_dtype_{f32,f64} for the mlx backend.
 *
 * mx::astype builds a new node in mlx's autograd graph; the
 * OP_CAST_DTYPE tape entry's scalar_arg encodes the target dtype for
 * replay (0.0 = f32, 1.0 = f64). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_cast_dtype_f32_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::astype(t->data, mx::float32), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 0.0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cast_dtype_f32(TensorHandle h) {
    return tensor_cast_dtype_f32_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_cast_dtype_f64_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::astype(t->data, mx::float64), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 1.0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cast_dtype_f64(TensorHandle h) {
    return tensor_cast_dtype_f64_mlx_streamed(h, default_stream_tag());
}

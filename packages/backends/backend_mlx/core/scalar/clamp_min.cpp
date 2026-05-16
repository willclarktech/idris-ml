/* tensor_clamp_min for the mlx backend. Implemented as mx::maximum
 * against a dtype-matched scalar; backward (via mx::vjp) zeros the
 * gradient at clamped indices. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_clamp_min_mlx_streamed(TensorHandle h, double min_val, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::maximum(t->data, scalar_like(min_val, t->data)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CLAMP_MIN, r, t, nullptr, min_val);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_clamp_min(TensorHandle h, double min_val) {
    return tensor_clamp_min_mlx_streamed(h, min_val, default_stream_tag());
}

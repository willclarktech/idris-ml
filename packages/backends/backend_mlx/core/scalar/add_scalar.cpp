/* tensor_add_scalar for the mlx backend. The scalar is built with
 * dtype matching the tensor (via precision.h's scalar_like) so an f64
 * tensor stays in f64. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_add_scalar_mlx_streamed(TensorHandle h, double s, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::add(t->data, scalar_like(s, t->data)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_ADD_SCALAR, r, t, nullptr, s);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_add_scalar(TensorHandle h, double s) {
    return tensor_add_scalar_mlx_streamed(h, s, default_stream_tag());
}

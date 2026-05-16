/* tensor_mul_scalar for the mlx backend. See add_scalar.cpp for the
 * dtype-matched scalar pattern. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_mul_scalar_mlx_streamed(TensorHandle h, double s, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::multiply(t->data, scalar_like(s, t->data)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MUL_SCALAR, r, t, nullptr, s);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_mul_scalar(TensorHandle h, double s) {
    return tensor_mul_scalar_mlx_streamed(h, s, default_stream_tag());
}

/* tensor_gelu for the mlx backend.
 *
 * tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
 * Decomposed into primitives so the tape captures each step. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_gelu_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto x = t->data;
    auto kGeluC  = scalar_like(0.7978845608028654, x);
    auto kGeluC3 = scalar_like(0.044715,           x);
    auto kThree  = scalar_like(3.0,                x);
    auto inner = mx::multiply(kGeluC, mx::add(x, mx::multiply(kGeluC3, mx::power(x, kThree))));
    auto result = mx::multiply(mx::multiply(half_like(x), x), mx::add(one_like(x), mx::tanh(inner)));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_GELU, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_gelu(TensorHandle h) {
    return tensor_gelu_mlx_streamed(h, default_stream_tag());
}

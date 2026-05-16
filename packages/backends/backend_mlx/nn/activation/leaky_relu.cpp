/* tensor_leaky_relu for the mlx backend.
 *
 * leaky_relu(x, alpha) = max(alpha*x, x). The alpha scalar lives on
 * the tape entry's scalar_arg for backward replay. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_leaky_relu_mlx_streamed(TensorHandle h, double alpha, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto alpha_arr = scalar_like(alpha, t->data);
    auto result = mx::maximum(mx::multiply(alpha_arr, t->data), t->data);
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_LEAKY_RELU, r, t, nullptr, alpha);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_leaky_relu(TensorHandle h, double alpha) {
    return tensor_leaky_relu_mlx_streamed(h, alpha, default_stream_tag());
}

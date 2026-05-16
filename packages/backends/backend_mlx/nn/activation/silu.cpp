/* tensor_silu (Swish) for the mlx backend.
 *
 * silu(x) = x * sigmoid(x). mlx has no native silu; this is the same
 * decomposition torch::silu does internally. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_silu_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto result = mx::multiply(t->data, mx::sigmoid(t->data));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_SILU, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_silu(TensorHandle h) {
    return tensor_silu_mlx_streamed(h, default_stream_tag());
}

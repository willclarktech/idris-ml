/* tensor_softmax + 2d/3d variants for the mlx backend.
 *
 * mx::softmax over the requested axis; OP_SOFTMAX_2D is reused as the
 * tape opcode for backward (rank-N case has the same vjp formula —
 * softmax * (g - sum(g * softmax, dim=keep))). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_softmax_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, dim), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_softmax(TensorHandle h, int dim) {
    return tensor_softmax_mlx_streamed(h, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_softmax_2d_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, -1), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_softmax_2d(TensorHandle h) {
    return tensor_softmax_2d_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_softmax_3d_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, -1), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_3D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_softmax_3d(TensorHandle h) {
    return tensor_softmax_3d_mlx_streamed(h, default_stream_tag());
}

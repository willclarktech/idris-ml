/* View ops — mlx.
 *
 *   - tensor_view_2d / tensor_view_2d_mlx_streamed   pick (row, col)
 *     from a 2D matrix via flatten + take. Returns a 0-dim tensor that
 *     shares storage with the parent; the result inherits requires_grad
 *     from the parent.
 *   - tensor_view_1d / tensor_view_1d_mlx_streamed   pick index `idx`
 *     from a 1D vector via mx::take.
 *
 * Both ops are persistent leaves from a Tensor-lifecycle perspective
 * (mlx's refcount + no_grad-driven sweep govern them); the streamed
 * variants thread the user's mlx stream tag through the underlying
 * mx::take call so the result is bound to the requested device.
 */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" {

TensorHandle tensor_view_2d_mlx_streamed(TensorHandle mat, int row, int col, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)mat;
    // Return a scalar tensor sharing the value
    int cols = t->data.shape(1);
    auto val = mx::take(mx::flatten(t->data), mx::array(row * cols + col));
    auto r = new Tensor(val, t->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_view_2d(TensorHandle mat, int row, int col) {
    return tensor_view_2d_mlx_streamed(mat, row, col, default_stream_tag());
}

TensorHandle tensor_view_1d_mlx_streamed(TensorHandle vec, int idx, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)vec;
    auto val = mx::take(t->data, mx::array(idx));
    auto r = new Tensor(val, t->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_view_1d(TensorHandle vec, int idx) {
    return tensor_view_1d_mlx_streamed(vec, idx, default_stream_tag());
}

} // extern "C"

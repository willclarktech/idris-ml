/* tensor_reshape + fixed-rank variants for the mlx backend.
 *
 * The 1d/2d/3d/4d wrappers route through the generic streamed entry
 * so OP_RESHAPE is the only shape op replayed in backward — backward
 * doesn't care about the rank label, only the source/target shapes. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_reshape_mlx_streamed(TensorHandle h, int* shape, int rank, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    mx::Shape sh(shape, shape + rank);
    auto r = new Tensor(mx::reshape(t->data, sh), t->requires_grad);
    if (t->requires_grad) tape_append(OP_RESHAPE, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    return tensor_reshape_mlx_streamed(h, shape, rank, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_1d_mlx_streamed(TensorHandle h, int n, int stream_tag) {
    int shape[] = {n};
    return tensor_reshape_mlx_streamed(h, shape, 1, stream_tag);
}

extern "C" TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
    return tensor_reshape_1d_mlx_streamed(h, n, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_2d_mlx_streamed(TensorHandle h, int rows, int cols, int stream_tag) {
    int shape[] = {rows, cols};
    return tensor_reshape_mlx_streamed(h, shape, 2, stream_tag);
}

extern "C" TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
    return tensor_reshape_2d_mlx_streamed(h, rows, cols, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_3d_mlx_streamed(TensorHandle h, int d0, int d1, int d2, int stream_tag) {
    int shape[] = {d0, d1, d2};
    return tensor_reshape_mlx_streamed(h, shape, 3, stream_tag);
}

extern "C" TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
    return tensor_reshape_3d_mlx_streamed(h, d0, d1, d2, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_4d_mlx_streamed(TensorHandle h, int d0, int d1, int d2, int d3, int stream_tag) {
    int shape[] = {d0, d1, d2, d3};
    return tensor_reshape_mlx_streamed(h, shape, 4, stream_tag);
}

extern "C" TensorHandle tensor_reshape_4d(TensorHandle h, int d0, int d1, int d2, int d3) {
    return tensor_reshape_4d_mlx_streamed(h, d0, d1, d2, d3, default_stream_tag());
}

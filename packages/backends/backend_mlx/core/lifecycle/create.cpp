/* tensor_create variants for the mlx backend.
 *
 * Per-dtype creators (_f32 / _f64) plus the legacy unsuffixed entry
 * (routes to _f32). mx_array_from_doubles handles the f64-passthrough
 * vs f32-narrow case at allocation. Each grad-requiring tensor appends
 * an OP_CONST tape entry so backward replay treats it as a leaf. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

static TensorHandle tensor_create_impl(double* data, int* shape, int rank, int requires_grad, mx::Dtype dt) {
    mx::Shape sh(shape, shape + rank);
    auto t = new Tensor(mx_array_from_doubles(data, sh, dt), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    return (TensorHandle)t;
}

extern "C" TensorHandle tensor_create_f32_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_impl(data, shape, rank, requires_grad, mx::float32);
}

extern "C" TensorHandle tensor_create_f32(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_f32_mlx_streamed(data, shape, rank, requires_grad, default_stream_tag());
}

extern "C" TensorHandle tensor_create_f64_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_impl(data, shape, rank, requires_grad, mx::float64);
}

extern "C" TensorHandle tensor_create_f64(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_f64_mlx_streamed(data, shape, rank, requires_grad, default_stream_tag());
}

extern "C" TensorHandle tensor_create_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_f32_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
}

extern "C" TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_mlx_streamed(data, shape, rank, requires_grad, default_stream_tag());
}

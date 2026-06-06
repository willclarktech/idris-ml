/* tensor_create_scalar variants for the mlx backend.
 *
 * Per-dtype creators (_f32 / _f64) plus the legacy unsuffixed entry
 * (routes to _f32, the historical default on mlx — Metal GPU's
 * f32-only constraint shaped this choice). Each grad-requiring scalar
 * appends an OP_CONST tape entry so backward replay treats it as a
 * leaf. Non-grad scalars stay non-persistent — freed by tape_reset at
 * optimizer_step. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

static TensorHandle tensor_create_scalar_impl(double value, int requires_grad, mx::Dtype dt) {
    auto t = new Tensor(mx::array(value, dt), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    return (TensorHandle)t;
}

extern "C" TensorHandle tensor_create_scalar_f32_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_impl(value, requires_grad, mx::float32);
}

extern "C" TensorHandle tensor_create_scalar_f32(double value, int requires_grad) {
    return tensor_create_scalar_f32_mlx_streamed(value, requires_grad, default_stream_tag());
}

extern "C" TensorHandle tensor_create_scalar_f64_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_impl(value, requires_grad, mx::float64);
}

extern "C" TensorHandle tensor_create_scalar_f64(double value, int requires_grad) {
    return tensor_create_scalar_f64_mlx_streamed(value, requires_grad, default_stream_tag());
}

extern "C" TensorHandle tensor_create_scalar_bf16_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_impl(value, requires_grad, mx::bfloat16);
}

extern "C" TensorHandle tensor_create_scalar_f16_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_impl(value, requires_grad, mx::float16);
}

extern "C" TensorHandle tensor_create_scalar_i32_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_impl(value, requires_grad, mx::int32);
}

/* Legacy unsuffixed: route to fp32 (current historical behavior on mlx). */
extern "C" TensorHandle tensor_create_scalar_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_f32_mlx_streamed(value, requires_grad, stream_tag);
}

extern "C" TensorHandle tensor_create_scalar(double value, int requires_grad) {
    return tensor_create_scalar_mlx_streamed(value, requires_grad, default_stream_tag());
}

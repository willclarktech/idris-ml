/* Persistent-leaf creators — mlx.
 *
 * Houses the base param + state creators (1d/2d/3d/4d for param, 1d/2d
 * for state) plus the related plain non-grad 1d/2d creators. All call
 * sites of the static `tensor_create_impl` helper live here, so the
 * helper moved with them — the monolith no longer references it.
 *
 *   - tensor_create_impl              builds an mx::array from a host
 *                                     double buffer, wraps it in a Tensor,
 *                                     and pushes OP_CONST onto the tape
 *                                     when requires_grad=true.
 *   - tensor_create_{1d,2d}_impl      thin wrappers fixing rank.
 *   - tensor_create_param_{1,2,3,4}d_impl
 *                                     same as above but with requires_grad
 *                                     forced to 1 + free(data) at the end
 *                                     because the param-creator FFI
 *                                     contract owns the host buffer.
 *   - tensor_create_state_{1,2}d_impl no requires_grad; persistent leaf.
 *
 * Each impl is fronted by an extern "C" `_f32_mlx_streamed`,
 * `_f64_mlx_streamed`, `_mlx_streamed`, plus the legacy unstreamed entry
 * the dispatcher in training/dtype_dispatch.cpp routes into.
 */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include <cstdlib>

static TensorHandle tensor_create_impl(double* data, int* shape, int rank, int requires_grad, mx::Dtype dt) {
    mx::Shape sh(shape, shape + rank);
    auto t = new Tensor(mx_array_from_doubles(data, sh, dt), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    return (TensorHandle)t;
}

/* ---------- Non-grad 1d / 2d creators ---------- */

// Internal: dtype-parameterized 1d/2d creators.
static TensorHandle tensor_create_1d_impl(int n, double* data, int requires_grad, mx::Dtype dt) {
    int shape[] = {n};
    auto t = tensor_create_impl(data, shape, 1, requires_grad, dt);
    free(data);
    return t;
}

static TensorHandle tensor_create_2d_impl(int rows, int cols, double* data, int requires_grad, mx::Dtype dt) {
    int shape[] = {rows, cols};
    auto t = tensor_create_impl(data, shape, 2, requires_grad, dt);
    free(data);
    return t;
}

extern "C" TensorHandle tensor_create_1d_f32_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_1d_impl(n, data, requires_grad, mx::float32);
}
extern "C" TensorHandle tensor_create_1d_f32(int n, double* data, int requires_grad) {
    return tensor_create_1d_f32_mlx_streamed(n, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_1d_f64_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_1d_impl(n, data, requires_grad, mx::float64);
}
extern "C" TensorHandle tensor_create_1d_f64(int n, double* data, int requires_grad) {
    return tensor_create_1d_f64_mlx_streamed(n, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_1d_bf16_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_1d_impl(n, data, requires_grad, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_1d_f16_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_1d_impl(n, data, requires_grad, mx::float16);
}
extern "C" TensorHandle tensor_create_1d_i32_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_1d_impl(n, data, requires_grad, mx::int32);
}
extern "C" TensorHandle tensor_create_2d_f32_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::float32);
}
extern "C" TensorHandle tensor_create_2d_f32(int rows, int cols, double* data, int requires_grad) {
    return tensor_create_2d_f32_mlx_streamed(rows, cols, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_2d_f64_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::float64);
}
extern "C" TensorHandle tensor_create_2d_f64(int rows, int cols, double* data, int requires_grad) {
    return tensor_create_2d_f64_mlx_streamed(rows, cols, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_2d_bf16_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_2d_f16_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::float16);
}
extern "C" TensorHandle tensor_create_2d_i32_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::int32);
}
extern "C" TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    return tensor_create_2d_f64(rows, cols, data, requires_grad);
}

/* ---------- 3d param creators ---------- */

static TensorHandle tensor_create_param_3d_impl(int d0, int d1, int d2, double* data, mx::Dtype dt) {
    int shape[] = {d0, d1, d2};
    auto t = tensor_create_impl(data, shape, 3, 1, dt);
    free(data);
    return t;
}
extern "C" TensorHandle tensor_create_param_3d_f32_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_impl(d0, d1, d2, data, mx::float32);
}
extern "C" TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* data) {
    return tensor_create_param_3d_f32_mlx_streamed(d0, d1, d2, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_3d_f64_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_impl(d0, d1, d2, data, mx::float64);
}
extern "C" TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* data) {
    return tensor_create_param_3d_f64_mlx_streamed(d0, d1, d2, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_3d_bf16_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_impl(d0, d1, d2, data, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_param_3d_f16_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_impl(d0, d1, d2, data, mx::float16);
}
extern "C" TensorHandle tensor_create_param_3d_i32_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_impl(d0, d1, d2, data, mx::int32);
}

/* ---------- 1d / 2d / 4d param creators ---------- */

static TensorHandle tensor_create_param_1d_impl(int n, double* data, mx::Dtype dt) {
    int shape[] = {n};
    auto t = tensor_create_impl(data, shape, 1, 1, dt);
    free(data);
    return t;
}
static TensorHandle tensor_create_param_2d_impl(int rows, int cols, double* data, mx::Dtype dt) {
    int shape[] = {rows, cols};
    auto t = tensor_create_impl(data, shape, 2, 1, dt);
    free(data);
    return t;
}
static TensorHandle tensor_create_param_4d_impl(int d0, int d1, int d2, int d3, double* data, mx::Dtype dt) {
    int shape[] = {d0, d1, d2, d3};
    auto t = tensor_create_impl(data, shape, 4, 1, dt);
    free(data);
    return t;
}

extern "C" TensorHandle tensor_create_param_1d_f32_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_1d_impl(n, data, mx::float32);
}
extern "C" TensorHandle tensor_create_param_1d_f32(int n, double* data) {
    return tensor_create_param_1d_f32_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_1d_f64_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_1d_impl(n, data, mx::float64);
}
extern "C" TensorHandle tensor_create_param_1d_f64(int n, double* data) {
    return tensor_create_param_1d_f64_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_1d_bf16_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_1d_impl(n, data, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_param_1d_f16_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_1d_impl(n, data, mx::float16);
}
extern "C" TensorHandle tensor_create_param_1d_i32_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_1d_impl(n, data, mx::int32);
}
extern "C" TensorHandle tensor_create_param_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_2d_impl(rows, cols, data, mx::float32);
}
extern "C" TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* data) {
    return tensor_create_param_2d_f32_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_2d_f64_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_2d_impl(rows, cols, data, mx::float64);
}
extern "C" TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* data) {
    return tensor_create_param_2d_f64_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_2d_bf16_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_2d_impl(rows, cols, data, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_param_2d_f16_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_2d_impl(rows, cols, data, mx::float16);
}
extern "C" TensorHandle tensor_create_param_2d_i32_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_2d_impl(rows, cols, data, mx::int32);
}
extern "C" TensorHandle tensor_create_param_4d_f32_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::float32);
}
extern "C" TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* data) {
    return tensor_create_param_4d_f32_mlx_streamed(d0, d1, d2, d3, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_4d_f64_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::float64);
}
extern "C" TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* data) {
    return tensor_create_param_4d_f64_mlx_streamed(d0, d1, d2, d3, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_4d_bf16_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_param_4d_f16_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::float16);
}
extern "C" TensorHandle tensor_create_param_4d_i32_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::int32);
}


/* ---------- 1d / 2d state creators ---------- */

static TensorHandle tensor_create_state_1d_impl(int n, double* data, mx::Dtype dt) {
    int shape[] = {n};
    auto t = tensor_create_impl(data, shape, 1, 0, dt);
    free(data);
    return t;
}
static TensorHandle tensor_create_state_2d_impl(int rows, int cols, double* data, mx::Dtype dt) {
    int shape[] = {rows, cols};
    auto t = tensor_create_impl(data, shape, 2, 0, dt);
    free(data);
    return t;
}

extern "C" TensorHandle tensor_create_state_1d_f32_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_1d_impl(n, data, mx::float32);
}
extern "C" TensorHandle tensor_create_state_1d_f32(int n, double* data) {
    return tensor_create_state_1d_f32_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_1d_f64_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_1d_impl(n, data, mx::float64);
}
extern "C" TensorHandle tensor_create_state_1d_f64(int n, double* data) {
    return tensor_create_state_1d_f64_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_1d_bf16_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_1d_impl(n, data, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_state_1d_f16_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_1d_impl(n, data, mx::float16);
}
extern "C" TensorHandle tensor_create_state_1d_i32_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_1d_impl(n, data, mx::int32);
}
extern "C" TensorHandle tensor_create_state_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_2d_impl(rows, cols, data, mx::float32);
}
extern "C" TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* data) {
    return tensor_create_state_2d_f32_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_2d_f64_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_2d_impl(rows, cols, data, mx::float64);
}
extern "C" TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* data) {
    return tensor_create_state_2d_f64_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_2d_bf16_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_2d_impl(rows, cols, data, mx::bfloat16);
}
extern "C" TensorHandle tensor_create_state_2d_f16_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_2d_impl(rows, cols, data, mx::float16);
}
extern "C" TensorHandle tensor_create_state_2d_i32_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_state_2d_impl(rows, cols, data, mx::int32);
}


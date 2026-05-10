/* shared/training/dtype_streamed.c — backend-agnostic dtag-streamed
 * create / cast wrappers.
 *
 * The 11 unified `tensor_create_*_streamed` FFI entry points (closed
 * 2026-05-22) plus `tensor_cast_dtype_streamed`. Each is a one-liner
 * that strips the (mlx-only) `stream_tag` parameter and forwards to
 * the corresponding `g_active_port` dtag dispatcher. The actual dtag
 * dispatch logic lives in each backend's adapter — for tape, see
 * backend_tape/training/dtype_dispatch.c.
 *
 * Compiled once per backend in TRAINING_ADAPTER_BACKENDS so multi-link
 * gets `tensor_create_*_streamed_<b>` exports.
 *
 * stream_tag rationale: mlx uses `stream_tag` to pick which compute
 * stream (CPU/GPU) the tensor lands on. Tape and torch ignore it
 * (single-stream backends). The shared wrappers absorb the parameter
 * at the boundary so the port surface stays simpler — when mlx's
 * adapter lands, it'll need a separate port method or carry the
 * stream tag in a thread-local.
 *
 * Lint exemption: hand-maintained (not manifest-generated) — see
 * reference_dtype_ffi_wrappers in user memory.
 */

#include "port.h"
#include "../../backend.h"

TensorHandle tensor_create_scalar_streamed(double v, int rg, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_scalar(v, rg, dtag);
}

TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int rg, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create(data, shape, rank, rg, dtag);
}

TensorHandle tensor_create_1d_streamed(int n, double* data, int rg, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_1d(n, data, rg, dtag);
}

TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int rg, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_2d(rows, cols, data, rg, dtag);
}

TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_param_1d(n, data, dtag);
}

TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_param_2d(rows, cols, data, dtag);
}

TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_param_3d(d0, d1, d2, data, dtag);
}

TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_param_4d(d0, d1, d2, d3, data, dtag);
}

TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_state_1d(n, data, dtag);
}

TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.create_state_2d(rows, cols, data, dtag);
}

TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag) {
    (void)stream_tag;
    return (TensorHandle)g_active_port.cast_dtype((void*)src, dtag);
}

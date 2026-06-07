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

TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int rg, int stream_tag,
                                    int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create(data, shape, rank, rg, dtag);
}

TensorHandle tensor_create_1d_streamed(int n, double* data, int rg, int stream_tag, int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_1d(n, data, rg, dtag);
}

TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int rg, int stream_tag,
                                       int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_2d(rows, cols, data, rg, dtag);
}

TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_param_1d(n, data, dtag);
}

TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag,
                                             int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_param_2d(rows, cols, data, dtag);
}

TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag,
                                             int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_param_3d(d0, d1, d2, data, dtag);
}

TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data,
                                             int stream_tag, int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_param_4d(d0, d1, d2, d3, data, dtag);
}

TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_state_1d(n, data, dtag);
}

TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag,
                                             int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.create_state_2d(rows, cols, data, dtag);
}

TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag) {
	(void)stream_tag;
	return (TensorHandle)g_active_port.cast_dtype((void*)src, dtag);
}

/* ---- Fused param create + init ----
 * Each `tensor_create_param_<rank>_<init>_streamed` forwards to the
 * port's create_param_<rank>_<init>. If the active backend hasn't
 * wired the slot (nullptr), abort loudly at the FFI boundary so the
 * failure mode is "missing backend support" not "silent miscall".
 * See the port.h struct doc for the per-backend wiring story. */
#include <stdio.h>
#include <stdlib.h>

static void abort_unwired_init(const char* fn) {
	fprintf(stderr,
	        "%s: this backend hasn't wired the fused-init port methods yet. "
	        "If you need this primitive, port the create_param_*_normal / "
	        "create_param_*_const slots in the backend's adapter (see torch's "
	        "adapter.cpp for the reference impl).\n",
	        fn);
	abort();
}

TensorHandle tensor_create_param_1d_normal_streamed(int n, double mean, double std, int stream_tag,
                                                    int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_1d_normal) abort_unwired_init("tensor_create_param_1d_normal");
	return (TensorHandle)g_active_port.create_param_1d_normal(n, mean, std, dtag);
}

TensorHandle tensor_create_param_2d_normal_streamed(int rows, int cols, double mean, double std,
                                                    int stream_tag, int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_2d_normal) abort_unwired_init("tensor_create_param_2d_normal");
	return (TensorHandle)g_active_port.create_param_2d_normal(rows, cols, mean, std, dtag);
}

TensorHandle tensor_create_param_3d_normal_streamed(int d0, int d1, int d2, double mean, double std,
                                                    int stream_tag, int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_3d_normal) abort_unwired_init("tensor_create_param_3d_normal");
	return (TensorHandle)g_active_port.create_param_3d_normal(d0, d1, d2, mean, std, dtag);
}

TensorHandle tensor_create_param_4d_normal_streamed(int d0, int d1, int d2, int d3, double mean,
                                                    double std, int stream_tag, int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_4d_normal) abort_unwired_init("tensor_create_param_4d_normal");
	return (TensorHandle)g_active_port.create_param_4d_normal(d0, d1, d2, d3, mean, std, dtag);
}

TensorHandle tensor_create_param_1d_const_streamed(int n, double value, int stream_tag, int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_1d_const) abort_unwired_init("tensor_create_param_1d_const");
	return (TensorHandle)g_active_port.create_param_1d_const(n, value, dtag);
}

TensorHandle tensor_create_param_2d_const_streamed(int rows, int cols, double value, int stream_tag,
                                                   int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_2d_const) abort_unwired_init("tensor_create_param_2d_const");
	return (TensorHandle)g_active_port.create_param_2d_const(rows, cols, value, dtag);
}

TensorHandle tensor_create_param_3d_const_streamed(int d0, int d1, int d2, double value,
                                                   int stream_tag, int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_3d_const) abort_unwired_init("tensor_create_param_3d_const");
	return (TensorHandle)g_active_port.create_param_3d_const(d0, d1, d2, value, dtag);
}

TensorHandle tensor_create_param_4d_const_streamed(int d0, int d1, int d2, int d3, double value,
                                                   int stream_tag, int dtag) {
	(void)stream_tag;
	if (!g_active_port.create_param_4d_const) abort_unwired_init("tensor_create_param_4d_const");
	return (TensorHandle)g_active_port.create_param_4d_const(d0, d1, d2, d3, value, dtag);
}

void tensor_set_init_seed_streamed(unsigned long long seed, int stream_tag) {
	(void)stream_tag;
	if (g_active_port.set_init_seed) g_active_port.set_init_seed((uint64_t)seed);
	/* else: silent no-op — some backends don't support init-RNG seeding (e.g. mlx). */
}

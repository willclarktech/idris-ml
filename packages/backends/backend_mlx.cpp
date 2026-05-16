/* backend_mlx.cpp — MLX backend implementing backend.h.
 *
 * Uses Apple's MLX framework for GPU-accelerated tensor operations
 * on Apple Silicon via Metal. Forward ops record to a tape; backward
 * replays the tape inside mlx::grad for native autograd — zero
 * hand-written backward rules.
 *
 * Build: make BACKEND=mlx MLX_SITE=/path/to/mlx backend
 */

#include "backend.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>      // _exit
#include <exception>     // std::set_terminate, std::terminate_handler
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <mlx/mlx.h>

#include "backend_mlx/tensor.h"
#include "backend_mlx/tape.h"
#include "backend_mlx/stream.h"
#include "backend_mlx/precision.h"
/* `namespace mx = mlx::core;` is provided by backend_mlx/tensor.h. */

/* Backend init constructor (mlx_backend_init + std::terminate gate) lives
 * in backend_mlx/init.cpp. */

/* ================================================================
   Per-call stream selection
   ================================================================
   `Tensor [..] (MlxDev MGpu)` and `Tensor [..] (MlxDev MCpu)` are
   distinct types in the Idris-side type system; the C-side runtime
   should honour that distinction by running each op on the right
   mlx stream. The `UserDeviceCore (MlxDev s)` instance derives an
   int stream tag from `s` (0 = cpu, 1 = gpu) and threads it through
   the `_streamed` FFI variants below. Each streamed entry opens an
   `mx::StreamContext` from the cached `cpu_stream` / `gpu_stream`,
   so the array's primitive ties to the chosen stream and mlx's
   autograd (`mx::vjp`) automatically replays the backward on the
   same stream.

   The legacy unstreamed entry points (the existing `tensor_*_mlx`
   symbols, aliased to unsuffixed names when mlx is primary) keep
   working as one-line trampolines: they invoke their `_streamed`
   counterpart with `default_stream_tag()` so behaviour matches the
   pre-streams world for callers that don't have a typed stream
   (smart constructors in `Tensor.idr`, layer-level direct calls
   in `Layer/Dnc.idr`, etc.). Threading the stream all the way to
   those callers requires a wider Idris refactor — separate row. */

/* cpu_stream / gpu_stream / stream_for_tag / default_stream_tag +
   the WITH_STREAM macro live in backend_mlx/stream.h. */

/* Forward declarations for `_mlx_streamed` symbols that are called from
   inside other `_mlx_streamed` bodies earlier in the file (the L60
   composition-fix threads stream_tag through every inner call). C++
   requires these to be declared before use; their definitions appear
   later. */
extern "C" {
/* core/elementwise/ + core/scalar/ extractions — declared here so
   internal callers (loss helpers, lstm gates, etc.) can reach the
   streamed variants now that they live in separate TUs. */
TensorHandle tensor_add_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
TensorHandle tensor_sub_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
TensorHandle tensor_mul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
TensorHandle tensor_div_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
TensorHandle tensor_neg_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_abs_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_exp_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_log_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_sqrt_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_pow_mlx_streamed(TensorHandle hbase, TensorHandle hexp, int stream_tag);
TensorHandle tensor_sigmoid_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_tanh_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_softplus_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_add_scalar_mlx_streamed(TensorHandle h, double s, int stream_tag);
TensorHandle tensor_mul_scalar_mlx_streamed(TensorHandle h, double s, int stream_tag);
TensorHandle tensor_clamp_min_mlx_streamed(TensorHandle h, double min_val, int stream_tag);
TensorHandle tensor_transpose_last2_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_bmm_3x3_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
TensorHandle tensor_masked_fill_mlx_streamed(TensorHandle h, TensorHandle hmask, double value, int stream_tag);
TensorHandle tensor_softmax_3d_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_reshape_mlx_streamed(TensorHandle h, int* shape, int rank, int stream_tag);
TensorHandle tensor_narrow_mlx_streamed(TensorHandle h, int dim, int start, int len, int stream_tag);
TensorHandle tensor_select(TensorHandle h, int dim, int index);
TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim);
TensorHandle tensor_create_scalar_f32_mlx_streamed(double value, int requires_grad, int stream_tag);
TensorHandle tensor_create_scalar_f64_mlx_streamed(double value, int requires_grad, int stream_tag);
TensorHandle tensor_create_scalar_mlx_streamed(double value, int requires_grad, int stream_tag);
TensorHandle tensor_create_f32_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
TensorHandle tensor_create_f64_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
TensorHandle tensor_create_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
TensorHandle tensor_cast_dtype_f32_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_cast_dtype_f64_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_clone_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_mean_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_mean(TensorHandle h);
double       tensor_item_mlx_streamed(TensorHandle h, int stream_tag);
double       tensor_item_1d_mlx_streamed(TensorHandle vec, int idx, int stream_tag);
void         tensor_free_mlx_streamed(TensorHandle h, int stream_tag);
TensorHandle tensor_create_param_1d_f32_mlx_streamed(int n, double* data, int stream_tag);
TensorHandle tensor_create_param_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag);
TensorHandle tensor_create_param_3d_f32_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag);
TensorHandle tensor_create_param_4d_f32_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag);
TensorHandle tensor_create_state_1d_f32_mlx_streamed(int n, double* data, int stream_tag);
TensorHandle tensor_create_state_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag);
}

/* ================================================================
   Hot-path scalar constants
   ================================================================
   Lazy-init via never-destroyed heap singletons. Sharing a constant
   across calls is safe — mlx arrays are immutable from an op's
   perspective; ops produce new arrays rather than mutating inputs.
   Avoid using these as the rhs of mx::outer or similar ops where
   persistent operands hit a documented slow path (see gotchas.md).

   Never-destroyed is intentional: an mx::array's destructor decrements
   a refcount on an mlx-allocator-owned buffer. At process exit, our
   compilation unit's static destructors race against mlx's internal
   statics; on Apple Virtualization VMs (Tart, GHA macOS) the allocator
   throws `[malloc] Unable to allocate N bytes` when its backing device
   is already gone. Leaking ~12 bytes + one tiny mlx buffer at exit is
   the right trade. */
/* kF32_ZERO / kF32_ONE / kF32_HALF + the dtype-matching helpers
   (scalar_like / zero_like / one_like / half_like) all live in
   backend_mlx/precision.h. */

/* ================================================================
   Stub macro
   ================================================================ */

#define STUB() do { \
    fprintf(stderr, "MLX backend: %s not implemented\n", __func__); \
    abort(); \
} while(0)

/* Tensor lifecycle core (the four tracking globals + Tensor::Tensor
 * ctor + tensor_retain_internal / tensor_release_internal +
 * tensor_retain_handle / tensor_release_handle) lives in
 * backend_mlx/core/lifecycle/lifecycle_core.cpp. */

/* Precision bridge — mx_to_doubles / mx_read_double / mx_from_doubles /
   mx_array_from_doubles + scalar_like + zero/one/half_like all live in
   backend_mlx/precision.h. */

/* ================================================================
   Tape — autograd Wengert list
   ================================================================ */

/* Tape mechanics (tape vector definition + prof_tape_appends_mlx counter +
   tape_append + tape_reset) all live in backend_mlx/tape.cpp.
   OP_* enum, *ReplayMeta structs, TapeEntry, and extern decls are in
   backend_mlx/tape.h. no_grad_depth_mlx is defined in
   backend_mlx/training/autograd.cpp (co-located with the begin/end
   mutators); tape_append reads it via tape.h's extern decl. */

/* tape_append + tape_reset live in backend_mlx/tape.cpp. */


/* ================================================================
   Parameter registry
   ================================================================ */

/* The parameter registry surface lives in shared/training/param_registry.c —
   see the deletion comment further down in this file for the rationale. */

/* Profiling counters + _wall_ms_mlx live in
   backend_mlx/training/profiling.{h,cpp}. */
#include "backend_mlx/training/profiling.h"

/* Lifecycle ops (tensor_create_scalar* / tensor_create* /
   tensor_cast_dtype_* / tensor_clone / tensor_free) extracted to
   backend_mlx/core/lifecycle/. Internal callers reach the _mlx_streamed
   variants via the forward-decl block earlier in this file.

   The base persistent-leaf creators (tensor_create_impl helper +
   tensor_create_1d / 2d / param_{1,2,3,4}d / state_{1,2}d × F32/F64/
   streamed pairs) live in
   backend_mlx/core/lifecycle/create_param_state.cpp. */


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

/* tensor_item_1d / tensor_item_2d extracted to
   backend_mlx/core/lifecycle/. */

/* The native optimizer surface (Optimizer + create/free/step/zero_grad +
   set_lr / set_param_lr + clip_grad_value/norm + polyak_blend +
   optimizer_buf_count + get/set m/v/meta + the optional MLX_OPT_COMPILE
   Adam compile path + native_train_step + optimizer_step_with_clip)
   lives in backend_mlx/training/optimizer.cpp. */

/* ================================================================
   System
   ================================================================ */

/* get_rss_mb / get_current_rss_mb live in shared_utils.c (compiled
 * once, unified symbol). Both are in the rename header's EXCLUDE
 * set so the unsuffixed references resolve to the shared TU. */
/* backend_memory_report / backend_supports_tensor_params removed
 * (no Idris-side callers). */
/* Backend meta (backend_name + backend_reset_for_eval + tensor_print +
 * tensor_mlx_compile_enabled / invocations / reset_stats +
 * g_compile_invocations) lives in backend_mlx/core/backend_meta.cpp.
 * backend_epoch_begin / backend_profile_reset / backend_profile_report
 * live in backend_mlx/training/profiling.cpp. */

} // extern "C"

/* ---------- Portable FFI helpers ---------- */

/* The 10 *_return / idrisml_seq helpers (tensor_backward_return,
   param_register_return, param_zero_all_grads_return,
   tensor_to_doubles_return, tensor_backward_conditional,
   tensor_backward_return_loss, idrisml_seq,
   backend_reset_for_eval_return, backend_profile_reset_return,
   backend_profile_report_return) live in shared/training/ffi_shims.c
   — see the matching deletion in backend_torch.cpp. native_train_step
   + optimizer_step_with_clip stay in-file: they use Optimizer-internal
   prefix-scoped clip variants (clip_grad_*_filtered) that the shared
   trampolines don't carry. */

extern "C" {
int tensor_live_count(int dummy) { (void)dummy; return (int)all_tensors.size(); }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_mlx_peak_live; }
/* dropout_random_seed lives in shared_utils.c. */
} // extern "C"


/* Dtag-keyed streamed creators + the mlx_dtype_unsupported abort live in
 * backend_mlx/training/dtype_dispatch.cpp. */


/* Shared training port adapter (mlx_port_* shims + g_active_port struct)
 * lives in backend_mlx/training/adapter.cpp. */


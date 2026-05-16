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

/* ================================================================
   Backend init — device selection
   ================================================================
   Default to CPU stream. mlx 0.31 GPU (Metal) on GH Actions macOS
   runners hits "Unable to allocate N bytes" for tiny allocations
   under sustained load (NTM/DNC scalar-heavy backward; CI run
   25457289084 on commit 1b8feff). Local Apple Silicon machines
   handle GPU fine, so users can opt in via `MLX_DEVICE=gpu`.

   See TODO.md "MLX backend: support CPU+f64 mode + dependent-types
   demo" for the proper device-aware Tensor parameterization. */
// Apple Virtualization VMs (Tart, GHA macOS runners) hit
// `std::runtime_error: [malloc] Unable to allocate N bytes` (tiny N, 4–512)
// during process *shutdown* on scalar-heavy workloads (NTM/DNC, SAC
// actor). Training completes cleanly and the profile report prints; the
// failure is in mlx-internal static destructors racing against the Metal
// device teardown. Synchronising + clearing caches before destructors
// fire (via atexit, which runs before destructors) doesn't help — the
// throwing destructor is inside mlx itself.
//
// Fix: gate `std::terminate` to swallow post-main exceptions. A flag set
// by atexit distinguishes "after main returned" from "during training" —
// real exceptions during training still abort normally.
static bool g_mlx_past_main = false;
static std::terminate_handler g_prev_terminate_handler = nullptr;

static void mlx_set_past_main(void) { g_mlx_past_main = true; }

static void mlx_terminate_handler(void) {
    if (g_mlx_past_main) {
        // Process already exited cleanly; this is a destructor-order
        // crash we can't fix without a libmlx-upstream change. Exit 0.
        _exit(0);
    }
    if (g_prev_terminate_handler) g_prev_terminate_handler();
    std::abort();
}

__attribute__((constructor))
static void mlx_backend_init(void) {
    const char* env = std::getenv("MLX_DEVICE");
    if (env && (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "metal") == 0)) {
        mx::set_default_device(mx::Device(mx::Device::gpu));
    } else {
        mx::set_default_device(mx::Device(mx::Device::cpu));
    }
    // Bump the allocator limit. mlx defaults this to 1.5× the Metal-reported
    // recommended working set size; on GH Actions macOS VMs that's tiny and
    // even the CPU stream's allocator inherits it, so heavy scalar-tensor
    // workloads (NTM/DNC backward) abort with "[malloc] Unable to allocate
    // N bytes". 16 GB is well above what any example needs and well below
    // the runner's RAM. Cache limit follows.
    // Leave memory_limit / cache_limit at mlx's defaults. The
    // "[malloc] Unable to allocate N bytes" failure on Apple
    // Virtualization VMs (Tart, GHA macOS) is *not* hit because of an
    // mlx limit — it's MetalAllocator throwing when paravirtualized
    // Metal refuses a new MTLBuffer (per-process resource limit, not
    // bytes). Stack trace confirms: throw originates in
    // MetalAllocator::malloc even when MLX_DEVICE=cpu, because on
    // Apple Silicon mlx routes all buffer allocations through Metal
    // (unified memory). The real fix is keeping live MTLBuffer count
    // low; see the refcount-driven Tensor lifecycle work.
    g_prev_terminate_handler = std::set_terminate(mlx_terminate_handler);
    std::atexit(mlx_set_past_main);
}

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

/* ================================================================
   Tensor representation
   ================================================================ */

/* Tensor struct + tracking globals + retain/release helpers are
   declared in backend_mlx/tensor.h (included above). The canonical
   definitions live here so the symbols have one home for the link.
   Per-op .cpp files in the modular tree see them via the header. */

std::vector<Tensor*> all_tensors;
std::vector<TensorPair*> all_pairs;
int next_pool_idx = 0;
long g_mlx_create_calls_global = 0;  /* monotonic Tensor-creation counter (feeds create_id) */
long g_mlx_peak_live = 0;            /* high-water mark of all_tensors.size() */

Tensor::Tensor(mx::array d, bool rg)
    : data(std::move(d)), grad(mx::array(0.0f)), requires_grad(rg),
      has_grad(false), tape_idx(-1),
      pool_idx(next_pool_idx++), refcount(0) {
    create_id = g_mlx_create_calls_global++;
    all_tensors.push_back(this);
    if ((long)all_tensors.size() > g_mlx_peak_live) g_mlx_peak_live = (long)all_tensors.size();
}

void tensor_retain_internal(Tensor* t) {
    if (t) t->refcount++;
}

void tensor_release_internal(Tensor* t) {
    if (t && t->refcount > 0) t->refcount--;
}

// C-exported retain/release for FFI consumers (Idris-side managed handles,
// Scheme guardian-drain callbacks).
extern "C" {
void tensor_retain_handle(void* h) {
    tensor_retain_internal(reinterpret_cast<Tensor*>(h));
}
void tensor_release_handle(void* h) {
    tensor_release_internal(reinterpret_cast<Tensor*>(h));
}
}  // extern "C"

/* Precision bridge — mx_to_doubles / mx_read_double / mx_from_doubles /
   mx_array_from_doubles + scalar_like + zero/one/half_like all live in
   backend_mlx/precision.h. */

/* ================================================================
   Tape — autograd Wengert list
   ================================================================ */

/* OP_* enum, *ReplayMeta structs, TapeEntry, tape vector +
   prof_tape_appends_mlx + tape_append declaration are all in
   backend_mlx/tape.h. Definitions live here for symbol uniqueness.
   no_grad_depth is defined in backend_mlx/training/autograd.cpp
   (co-located with the begin/end mutators); we read it via tape.h's
   extern decl. */

std::vector<TapeEntry> tape;
long prof_tape_appends_mlx = 0;

int tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
    if (no_grad_depth_mlx > 0) {
        if (result) {
            result->requires_grad = false;
            result->tape_idx = -1;
        }
        return -1;
    }
    int idx = (int)tape.size();
    tape.push_back({op, result, arg1, arg2, scalar_arg, nullptr});
    result->tape_idx = idx;
    // The tape holds args until tape_reset; retain them while it does.
    // Also retain the result — the FFI wrapper's wrap-and-retain holds
    // refcount=1, but the tape entry holding `result` is a second
    // long-term holder that must be reflected in the count, or backward
    // replay can see a freed Tensor when the Idris wrap dies + drain
    // releases before tape_reset.
    tensor_retain_internal(result);
    tensor_retain_internal(arg1);
    tensor_retain_internal(arg2);
    prof_tape_appends_mlx++;
    return idx;
}

void tape_reset() {
    // Force evaluation of all pending lazy ops first. Survivors may have
    // mx::array graphs that reference soon-to-be-freed state Tensors;
    // materializing those graphs now means the freed mx::array's
    // refcounted impl gets dropped cleanly via mlx's internal accounting
    // rather than dangling.
    {
        std::vector<mx::array> to_eval;
        for (auto* t : all_tensors) {
            to_eval.push_back(t->data);
            if (t->has_grad) to_eval.push_back(t->grad);
        }
        if (!to_eval.empty()) mx::eval(to_eval);
    }
    // Release the args + result we retained in tape_append. Must
    // happen before tape.clear() — once entries are gone we can't
    // find which Tensors we retained.
    for (auto& e : tape) {
        tensor_release_internal(e.result);
        tensor_release_internal(e.arg1);
        tensor_release_internal(e.arg2);
    }
    // Free op metadata
    for (auto& e : tape) {
        if (e.op == OP_LAYER_NORM_2D && e.meta) {
            delete (LayerNormReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_GRU_CELL && e.meta) {
            delete (GruCellReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_STACK && e.meta) {
            delete (std::vector<int>*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_CAT_MULTI && e.meta) {
            delete (std::vector<int>*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_TILE_2D && e.meta) {
            std::free(e.meta);
            e.meta = nullptr;
        }
        if (e.op == OP_BATCH_NORM && e.meta) {
            delete (BatchNormReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_CONV1D && e.meta) {
            delete (Conv1DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_MAX_POOL1D && e.meta) {
            delete (MaxPool1DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_CONV2D && e.meta) {
            delete (Conv2DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_CONV2D_BATCHED && e.meta) {
            delete (Conv2DBatchedReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_MAX_POOL2D && e.meta) {
            delete (MaxPool2DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_MAX_POOL2D_BATCHED && e.meta) {
            delete (MaxPool2DBatchedReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_SUM_DIM && e.meta) {
            delete (SumDimReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_LINEAR_2D && e.meta) {
            delete (LinearReplayMeta*)e.meta;
            e.meta = nullptr;
        }
    }
    tape.clear();
    // Refcount-driven sweep: delete Tensors whose count is 0 (no
    // long-term holder left — no Idris wrap, no tape entry, no
    // param_registry entry). Everything else stays.
    std::vector<Tensor*> survivors;
    for (auto* t : all_tensors) {
        if (t->refcount > 0) survivors.push_back(t);
        else delete t;
    }
    all_tensors = std::move(survivors);
    // Reassign pool indices to be contiguous (keeps pool vector compact)
    next_pool_idx = 0;
    for (auto* t : all_tensors) t->pool_idx = next_pool_idx++;
    // Free TensorPair structs
    for (auto* p : all_pairs) free(p);
    all_pairs.clear();
    // Hand cached buffers back to the OS each epoch. Without this, mlx's
    // cache holds onto buffers from the just-collected non-persistent
    // tensors, and on GH Actions macOS-latest VMs the cache hits its
    // (Metal-derived) limit fast enough to abort small allocations like
    // `[malloc] Unable to allocate 4 bytes`. Locally on M-series the
    // cache is fine; the call is cheap either way.
    mx::clear_cache();
}

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

   `tensor_create_impl` stays here too — duplicated as a TU-local
   static helper because the monolith's remaining param/state creators
   (tensor_create_param_* etc., not yet extracted) call it directly.
   Phase 6f will retire this once those creators move. */

static TensorHandle tensor_create_impl(double* data, int* shape, int rank, int requires_grad, mx::Dtype dt) {
    mx::Shape sh(shape, shape + rank);
    auto t = new Tensor(mx_array_from_doubles(data, sh, dt), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    return (TensorHandle)t;
}

extern "C" {

/* ================================================================
   Accessors
   ================================================================ */

/* tensor_item extracted to backend_mlx/core/lifecycle/item.cpp. */

int tensor_numel(TensorHandle h) { return (int)((Tensor*)h)->data.size(); }
int tensor_dim(TensorHandle h) { return (int)((Tensor*)h)->data.ndim(); }
int tensor_size(TensorHandle h, int dim) { return (int)((Tensor*)h)->data.shape(dim); }

void tensor_to_doubles(TensorHandle h, double* out) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    mx_to_doubles(t->data, out);
}

// Byte-level I64 readout — declared in backend.h with the byte-exact
// contract honoured only on backends with native int64 storage. mlx
// stores only F32/F64; integer storage round-trips through `double`,
// inheriting the same 2^53 ceiling as the lingua-franca path.
// Practically the safetensors I/O caller only reaches this on tensors
// already typed I64, which mlx can't construct (Compatible MlxDev I64
// is closed). Implemented for symbol completeness.
void tensor_to_int64(TensorHandle h, int64_t* out) {
    auto t = (Tensor*)h;
    int n = (int)t->data.size();
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    if (!tmp) return;
    mx::eval(t->data);
    mx_to_doubles(t->data, tmp);
    for (int i = 0; i < n; i++) out[i] = (int64_t)tmp[i];
    free(tmp);
}

void tensor_to_floats(TensorHandle h, float* out) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    int n = (int)t->data.size();
    if (t->data.dtype() == mx::float32) {
        const float* src = t->data.data<float>();
        for (int i = 0; i < n; i++) out[i] = src[i];
    } else {
        const double* src = t->data.data<double>();
        for (int i = 0; i < n; i++) out[i] = (float)src[i];
    }
}

const char* tensor_dtype_name(TensorHandle h) {
    auto t = (Tensor*)h;
    return (t->data.dtype() == mx::float32) ? "F32" : "F64";
}

/* ================================================================
   Arithmetic
   ================================================================ */

/* tensor_add / tensor_sub / tensor_mul / tensor_div + their _mlx_streamed
   variants extracted to backend_mlx/core/elementwise/. */

/* tensor_neg / abs / exp / log / sqrt / pow / sigmoid extracted to
   backend_mlx/core/elementwise/. */
/* tensor_gelu / tensor_leaky_relu / tensor_silu live in
   backend_mlx/nn/activation/. tensor_tanh lives in
   backend_mlx/core/elementwise/tanh.cpp. */

/* tensor_softplus extracted to backend_mlx/core/elementwise/softplus.cpp.
   tensor_add_scalar / tensor_mul_scalar / tensor_clamp_min extracted to
   backend_mlx/core/scalar/. */

/* Reduction ops live in backend_mlx/linear/reduction/ */

/* Linear algebra ops (matmul, mv, mm, linear, linear_2d, dot, outer,
 * bmm, bmm_3x3, transpose_2d, transpose_last2, tile_2d) live in
 * backend_mlx/linear/linalg/. */

/* Softmax / log_softmax (incl. _2d / _3d variants) live in
 * backend_mlx/nn/softmax/. Loss ops (bce_with_logits, cross_entropy,
 * mse_loss) live in backend_mlx/nn/loss/. cosine_similarity lives in
 * backend_mlx/nn/attention/. */

/* batch_norm / group_norm / dropout / layer_norm live in
 * backend_mlx/nn/norm/. cross_attention / embedding live in
 * backend_mlx/nn/attention/. gru_cell lives in
 * backend_mlx/nn/recurrent/. */

/* Index ops (gather, scatter_add) live in backend_mlx/linear/index/.
 * Sort ops (argsort, cumprod) live in backend_mlx/linear/sort/.
 *
 * Conv / pool ops (conv1d/2d + circular + transpose + grouped +
 * max_pool1d/2d[_batched] + avg_pool1d/2d) live in
 * backend_mlx/conv/. */


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
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* data) {
    return tensor_create_param_3d_f32_mlx_streamed(d0, d1, d2, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_3d_f64_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_impl(d0, d1, d2, data, mx::float64);
}
TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* data) {
    return tensor_create_param_3d_f64_mlx_streamed(d0, d1, d2, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_3d_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_param_3d_f32_mlx_streamed(d0, d1, d2, data, stream_tag);
}
TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    return tensor_create_param_3d_mlx_streamed(d0, d1, d2, data, default_stream_tag());
}


/* ================================================================
   Shape manipulation
   ================================================================ */

/* Shape ops (reshape*, squeeze, unsqueeze, select) live in
 * backend_mlx/linear/shape/. */

/* tensor_stack / tensor_cat / tensor_cat2 live in
 * backend_mlx/linear/concat/{stack,cat}.cpp. */

TensorHandle tensor_batch(TensorHandle* handles, int count) {
    /* Batch [...] tensors -> [count, ...] = stack along new dim 0 */
    return tensor_stack(handles, count, 0);
}
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
    auto t = (Tensor*)h;
    int B = (int)t->data.shape(0);
    *out_count = B;
    auto* arr = (TensorHandle*)malloc(B * sizeof(TensorHandle));
    /* tensor_select picks dim=0 index=i and removes that dim — that is exactly
       one slice of the unbatched output. OP_SELECT is already replayed at dim=0,
       so backward replay reconstructs the same gathers. */
    for (int i = 0; i < B; i++) {
        arr[i] = tensor_select((TensorHandle)t, 0, i);
    }
    return arr;
}

TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
    // Create one-hot encoded 1D tensor in the requested dtype so the result
    // honestly matches the Idris `dt` (0/1 is exact in every dtype). mlx
    // admits F32/F64 only (Metal-F32, CPU-F64) per the Compatible table;
    // under the kind-major dtag layout dtag 15 = F64, dtag 14 = F32. Any
    // other dtag would fail the Compatible gate Idris-side; this routes to
    // F32 as a sentinel so a stray call doesn't silently return F64.
    int total = n_tokens * vocab_size;
    std::vector<double> data(total, 0.0);
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            data[i * vocab_size + tok] = 1.0;
    }
    mx::Shape sh = {total};
    mx::Dtype dt = (dtag == 15) ? mx::float64 : mx::float32;
    auto t = new Tensor(mx_array_from_doubles(data.data(), sh, dt), false);
    free(tokens);
    return (TensorHandle)t;
}

/* ================================================================
   Autograd — replay-based native backward via mlx::grad

   Forward ops record to the tape. tensor_backward replays the tape
   inside a closure and passes it to mlx::grad for native autograd.
   Zero hand-written backward rules.
   ================================================================ */

/* Job 3 Phase B — compile-path probe counter. Defined here so
   tensor_backward (below) and the FFI getters (further down) share
   the same TU-local symbol. Real mx::compile wiring lands in a
   later stage. */
int g_compile_invocations = 0;

/* tensor_backward (replay-based backward) lives in
 * backend_mlx/training/backward.cpp. */

/* Autograd surface (tensor_grad/zero_grad/detach/with_grad/set_requires_grad +
 * tensor_no_grad_begin/end + tensor_epoch_begin/end + mlx_sweep_generation)
 * lives in backend_mlx/training/autograd.cpp. */


/* ================================================================
   Device
   ================================================================ */

TensorHandle tensor_to_device(TensorHandle t, const char* device) { return t; }
const char* tensor_device(TensorHandle t) { return "gpu"; }

/* LSTM ops (lstm_cell, lstm_gates, lstm_gates_pair) + TensorPair
 * accessor helpers (pair_first/second/free) live in
 * backend_mlx/nn/recurrent/. */

/* ================================================================
   Parameter registry — surface lifted into shared/training/param_registry.c.

   Routes through `g_active_port_mlx` for per-tensor accesses (numel,
   grad-read/write, zero, bulk load). The shared registry's
   tensor_retain/release wrap each register/clear so mlx's refcount
   lifecycle (where the registry contributes +1 to keep params alive
   against the all_tensors sweep) is preserved. The shared param_clear
   no longer triggers tape_reset — that's covered by optimizer_step
   and backend_reset_for_eval. */

/* tensor_subtract_scalar_inplace lives in
   backend_mlx/training/ntm_specific.cpp. */

/* ================================================================
   Convenience functions
   ================================================================ */

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

// Per-dtype exports.
extern "C" TensorHandle tensor_create_1d_f32_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_1d_impl(n, data, requires_grad, mx::float32);

}
TensorHandle tensor_create_1d_f32(int n, double* data, int requires_grad) {
    return tensor_create_1d_f32_mlx_streamed(n, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_1d_f64_mlx_streamed(int n, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_1d_impl(n, data, requires_grad, mx::float64);

}
TensorHandle tensor_create_1d_f64(int n, double* data, int requires_grad) {
    return tensor_create_1d_f64_mlx_streamed(n, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_2d_f32_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::float32);

}
TensorHandle tensor_create_2d_f32(int rows, int cols, double* data, int requires_grad) {
    return tensor_create_2d_f32_mlx_streamed(rows, cols, data, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_2d_f64_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_2d_impl(rows, cols, data, requires_grad, mx::float64);

}
TensorHandle tensor_create_2d_f64(int rows, int cols, double* data, int requires_grad) {
    return tensor_create_2d_f64_mlx_streamed(rows, cols, data, requires_grad, default_stream_tag());
}

// Legacy unsuffixed.
TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    return tensor_create_1d_f32(n, data, requires_grad);
}
TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    return tensor_create_2d_f32(rows, cols, data, requires_grad);
}

/* tensor_alloc_doubles / tensor_free_doubles / tensor_read_double /
 * tensor_ptr_array_alloc live in shared_utils.c. */

/* tensor_stack_from_array / tensor_cat_from_array live in
 * backend_mlx/linear/concat/{stack,cat}.cpp. */

/* ================================================================
   Tensor-level parameter creation
   ================================================================ */

// Internal: dtype-parameterized param creators.
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

// Per-dtype exports.
extern "C" TensorHandle tensor_create_param_1d_f32_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_1d_impl(n, data, mx::float32);

}
TensorHandle tensor_create_param_1d_f32(int n, double* data) {
    return tensor_create_param_1d_f32_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_1d_f64_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_1d_impl(n, data, mx::float64);

}
TensorHandle tensor_create_param_1d_f64(int n, double* data) {
    return tensor_create_param_1d_f64_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_2d_impl(rows, cols, data, mx::float32);

}
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* data) {
    return tensor_create_param_2d_f32_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_2d_f64_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_2d_impl(rows, cols, data, mx::float64);

}
TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* data) {
    return tensor_create_param_2d_f64_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_4d_f32_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::float32);

}
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* data) {
    return tensor_create_param_4d_f32_mlx_streamed(d0, d1, d2, d3, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_4d_f64_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_4d_impl(d0, d1, d2, d3, data, mx::float64);

}
TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* data) {
    return tensor_create_param_4d_f64_mlx_streamed(d0, d1, d2, d3, data, default_stream_tag());
}

// Legacy unsuffixed.
extern "C" TensorHandle tensor_create_param_2d_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_2d_f32_mlx_streamed(rows, cols, data, stream_tag);

}
TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    return tensor_create_param_2d_mlx_streamed(rows, cols, data, default_stream_tag());
}
TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    return tensor_create_param_4d_f32_mlx_streamed(d0, d1, d2, d3, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_param_1d_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_param_1d_f32_mlx_streamed(n, data, stream_tag);

}
TensorHandle tensor_create_param_1d(int n, double* data) {
    return tensor_create_param_1d_mlx_streamed(n, data, default_stream_tag());
}

// State Tensors — covers both init-time permanent state (NTM mask, batch
// norm running stats, transformer positional encoding, DNC mask) AND
// per-sequence transient state (Ntm/Dnc zeroState). Both flow through
// the same refcount-driven lifecycle. The Idris-side wrap-and-retain on
// creation gives the first refcount bump. "Permanent" state stays alive
// via the Idris model record holding its wrap for the whole training
// run; "transient" per-sequence state's wrap dies at the end of each
// forward and drain + sweep eventually frees it.
// Internal: dtype-parameterized state creators.
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

// Per-dtype exports.
extern "C" TensorHandle tensor_create_state_1d_f32_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_state_1d_impl(n, data, mx::float32);

}
TensorHandle tensor_create_state_1d_f32(int n, double* data) {
    return tensor_create_state_1d_f32_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_1d_f64_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_state_1d_impl(n, data, mx::float64);

}
TensorHandle tensor_create_state_1d_f64(int n, double* data) {
    return tensor_create_state_1d_f64_mlx_streamed(n, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_state_2d_impl(rows, cols, data, mx::float32);

}
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* data) {
    return tensor_create_state_2d_f32_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_2d_f64_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_state_2d_impl(rows, cols, data, mx::float64);

}
TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* data) {
    return tensor_create_state_2d_f64_mlx_streamed(rows, cols, data, default_stream_tag());
}

// Legacy unsuffixed.
extern "C" TensorHandle tensor_create_state_2d_mlx_streamed(int rows, int cols, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_state_2d_f32_mlx_streamed(rows, cols, data, stream_tag);

}
TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    return tensor_create_state_2d_mlx_streamed(rows, cols, data, default_stream_tag());
}
extern "C" TensorHandle tensor_create_state_1d_mlx_streamed(int n, double* data, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_state_1d_f32_mlx_streamed(n, data, stream_tag);

}
TensorHandle tensor_create_state_1d(int n, double* data) {
    return tensor_create_state_1d_mlx_streamed(n, data, default_stream_tag());
}

extern "C" TensorHandle tensor_view_2d_mlx_streamed(TensorHandle mat, int row, int col, int stream_tag) {
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

extern "C" TensorHandle tensor_view_1d_mlx_streamed(TensorHandle vec, int idx, int stream_tag) {
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
void backend_reset_for_eval(void) {
    tape_reset();
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* p_tensor = (Tensor*)param_tensor(i_);
        p_tensor->tape_idx = -1;
        p_tensor->has_grad = false;
        tape_append(OP_CONST, p_tensor, nullptr, nullptr, 0);
    }
}
/* backend_epoch_begin / backend_profile_reset / backend_profile_report
   live in backend_mlx/training/profiling.cpp. */

/* ================================================================
   Debug
   ================================================================ */

const char* backend_name(void) { return "mlx"; }

void tensor_print(TensorHandle h) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    std::cout << t->data << std::endl;
}

/* ================================================================
   MLX compile integration (Job 3 Phase B)
   ================================================================ */

int tensor_mlx_compile_enabled(void) {
    const char* v = std::getenv("MLX_COMPILE");
    if (!v) return 0;
    if (v[0] == '1' && v[1] == '\0') return 1;
    if (std::strcmp(v, "true") == 0) return 1;
    if (std::strcmp(v, "yes") == 0) return 1;
    return 0;
}

/* Counter g_compile_invocations defined near top of file (before
   tensor_backward, which references it). Getter/setter exposed here as
   part of the public FFI surface. */
int  tensor_mlx_compile_invocations(void) { return g_compile_invocations; }
void tensor_mlx_compile_reset_stats(void) { g_compile_invocations = 0; }

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

int tensor_live_count(int dummy) { (void)dummy; return (int)all_tensors.size(); }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_mlx_peak_live; }
/* dropout_random_seed lives in shared_utils.c. */

} // extern "C"


/* Dtag-keyed streamed creators + the mlx_dtype_unsupported abort live in
 * backend_mlx/training/dtype_dispatch.cpp. */


/* Shared training port adapter (mlx_port_* shims + g_active_port struct)
 * lives in backend_mlx/training/adapter.cpp. */


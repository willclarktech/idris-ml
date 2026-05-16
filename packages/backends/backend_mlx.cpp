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
TensorHandle tensor_create_scalar_f32_mlx_streamed(double value, int requires_grad, int stream_tag);
TensorHandle tensor_create_f32_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
TensorHandle tensor_create_param_1d_f32_mlx_streamed(int n, double* data, int stream_tag);
TensorHandle tensor_create_param_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag);
TensorHandle tensor_create_param_3d_f32_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag);
TensorHandle tensor_create_param_4d_f32_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag);
TensorHandle tensor_create_state_1d_f32_mlx_streamed(int n, double* data, int stream_tag);
TensorHandle tensor_create_state_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag);
TensorHandle tensor_clone_mlx_streamed(TensorHandle h, int stream_tag);
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
namespace {
inline const mx::array& kF32_ZERO() { static const mx::array* v = new mx::array(0.0f, mx::float32); return *v; }
inline const mx::array& kF32_ONE()  { static const mx::array* v = new mx::array(1.0f, mx::float32); return *v; }
inline const mx::array& kF32_HALF() { static const mx::array* v = new mx::array(0.5f, mx::float32); return *v; }
}

/* Dtype-matching helpers (scalar_like / zero_like / one_like / half_like)
   live in backend_mlx/precision.h alongside the precision-bridge ones. */

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

/* OP_* enum, *ReplayMeta structs, TapeEntry, tape vector + no_grad_depth
   + prof_tape_appends_mlx, tape_append declaration are all in
   backend_mlx/tape.h. Definitions live here for symbol uniqueness. */

std::vector<TapeEntry> tape;
int no_grad_depth = 0;
long prof_tape_appends_mlx = 0;

int tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
    if (no_grad_depth > 0) {
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

static void tape_reset() {
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

/* Profiling counters */
static double prof_backward_ms_mlx = 0, prof_optimizer_ms_mlx = 0;
static double prof_optimizer_math_ms_mlx = 0;
static int prof_epochs_mlx = 0;
/* prof_tape_appends_mlx is declared earlier (before tape_append uses it). */

static double _wall_ms_mlx(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ================================================================
   Lifecycle
   ================================================================ */

extern "C" {

// Internal impl: creation parameterized by dtype.
static TensorHandle tensor_create_scalar_impl(double value, int requires_grad, mx::Dtype dt) {
    auto t = new Tensor(mx::array(value, dt), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    // Non-grad scalars stay non-persistent — freed by tape_reset at optimizer_step
    return (TensorHandle)t;
}

static TensorHandle tensor_create_impl(double* data, int* shape, int rank, int requires_grad, mx::Dtype dt) {
    mx::Shape sh(shape, shape + rank);
    auto t = new Tensor(mx_array_from_doubles(data, sh, dt), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    // Non-grad data tensors: non-persistent, freed by tape_reset at optimizer_step
    return (TensorHandle)t;
}

// Per-dtype exports.
extern "C" TensorHandle tensor_create_scalar_f32_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_scalar_impl(value, requires_grad, mx::float32);

}
TensorHandle tensor_create_scalar_f32(double value, int requires_grad) {
    return tensor_create_scalar_f32_mlx_streamed(value, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_scalar_f64_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_scalar_impl(value, requires_grad, mx::float64);

}
TensorHandle tensor_create_scalar_f64(double value, int requires_grad) {
    return tensor_create_scalar_f64_mlx_streamed(value, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_f32_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_impl(data, shape, rank, requires_grad, mx::float32);

}
TensorHandle tensor_create_f32(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_f32_mlx_streamed(data, shape, rank, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_f64_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);

    return tensor_create_impl(data, shape, rank, requires_grad, mx::float64);

}
TensorHandle tensor_create_f64(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_f64_mlx_streamed(data, shape, rank, requires_grad, default_stream_tag());
}

// Legacy unsuffixed: route to fp32 (current historical behavior on mlx).
// Both have streamed counterparts (`*_mlx_streamed`) used by
// `UserDeviceCore (MlxDev s)` to honour the type-level stream tag;
// the unstreamed entry points trampoline to them with the global
// default tag so smart constructors / direct prim__ callers in
// `Tensor.idr` / `Layer/*` keep their current behaviour.
extern "C" TensorHandle tensor_create_scalar_mlx_streamed(double value, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_scalar_f32_mlx_streamed(value, requires_grad, stream_tag);
}
TensorHandle tensor_create_scalar(double value, int requires_grad) {
    return tensor_create_scalar_mlx_streamed(value, requires_grad, default_stream_tag());
}
extern "C" TensorHandle tensor_create_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    return tensor_create_f32_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
}
TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_mlx_streamed(data, shape, rank, requires_grad, default_stream_tag());
}

// Per-dtype cast primitives. mx::astype builds a new node in mlx's
// autograd graph; the OP_CAST_DTYPE tape entry's scalar_arg encodes
// the target dtype for replay (0.0 = f32, 1.0 = f64).
extern "C" TensorHandle tensor_cast_dtype_f32_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::astype(t->data, mx::float32), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 0.0);
    return (TensorHandle)r;

}
TensorHandle tensor_cast_dtype_f32(TensorHandle h) {
    return tensor_cast_dtype_f32_mlx_streamed(h, default_stream_tag());
}
extern "C" TensorHandle tensor_cast_dtype_f64_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::astype(t->data, mx::float64), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 1.0);
    return (TensorHandle)r;

}
TensorHandle tensor_cast_dtype_f64(TensorHandle h) {
    return tensor_cast_dtype_f64_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_clone_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto c = new Tensor(mx::array(t->data), false);
    return (TensorHandle)c;
}
TensorHandle tensor_clone(TensorHandle h) {
    return tensor_clone_mlx_streamed(h, default_stream_tag());
}

extern "C" void tensor_free_mlx_streamed(TensorHandle h, int stream_tag) {
    (void)stream_tag;  // no kernel — pure C-side bookkeeping
    if (!h) return;
    auto t = (Tensor*)h;
    // Skip registered params — they're managed by param_clear.
    for (int i_ = 0; i_ < param_count(); i_++) {
        if ((Tensor*)param_tensor(i_) == t) return;
    }
    // Refcount-driven world (since commit 7eab36c): forcing `delete t` here
    // leaves dangling Tensor* pointers in tape entries that still reference
    // this result/arg, and the next tape_reset crashes when it walks the
    // tape to release retains. Instead drop the caller's implicit hold;
    // the tape's own retains (set by tape_append on result/arg1/arg2) keep
    // the Tensor alive until tape_reset releases them and sweeps refcount=0.
    //
    // We must also defend against the caller passing a handle that was
    // already swept by a prior tape_reset (common when optimizer_step
    // ran between the user's create and their free): touching `t` would
    // be use-after-free. Probe all_tensors first; skip if absent.
    for (auto* alive : all_tensors) {
        if (alive == t) { tensor_release_internal(t); return; }
    }
}
void tensor_free(TensorHandle h) {
    tensor_free_mlx_streamed(h, default_stream_tag());
}

/* ================================================================
   Accessors
   ================================================================ */

extern "C" double tensor_item_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    mx::eval(t->data);
    if (t->data.dtype() == mx::float64) return t->data.item<double>();
    return (double)t->data.item<float>();
}
double tensor_item(TensorHandle h) {
    return tensor_item_mlx_streamed(h, default_stream_tag());
}

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
extern "C" TensorHandle tensor_gelu_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    // GELU tanh approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    auto x = t->data;
    auto kGeluC  = scalar_like(0.7978845608028654, x);
    auto kGeluC3 = scalar_like(0.044715,           x);
    auto kThree  = scalar_like(3.0,                x);
    auto inner = mx::multiply(kGeluC, mx::add(x, mx::multiply(kGeluC3, mx::power(x, kThree))));
    auto result = mx::multiply(mx::multiply(half_like(x), x), mx::add(one_like(x), mx::tanh(inner)));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_GELU, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_gelu(TensorHandle h) {
    return tensor_gelu_mlx_streamed(h, default_stream_tag());
}

/* tensor_tanh extracted to backend_mlx/core/elementwise/tanh.cpp. */

extern "C" TensorHandle tensor_leaky_relu_mlx_streamed(TensorHandle h, double alpha, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto alpha_arr = scalar_like(alpha, t->data);
    auto result = mx::maximum(mx::multiply(alpha_arr, t->data), t->data);
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_LEAKY_RELU, r, t, nullptr, alpha);
    return (TensorHandle)r;

}
TensorHandle tensor_leaky_relu(TensorHandle h, double alpha) {
    return tensor_leaky_relu_mlx_streamed(h, alpha, default_stream_tag());
}

extern "C" TensorHandle tensor_silu_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto result = mx::multiply(t->data, mx::sigmoid(t->data));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_SILU, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_silu(TensorHandle h) {
    return tensor_silu_mlx_streamed(h, default_stream_tag());
}

/* tensor_softplus extracted to backend_mlx/core/elementwise/softplus.cpp.
   tensor_add_scalar / tensor_mul_scalar / tensor_clamp_min extracted to
   backend_mlx/core/scalar/. */

/* ================================================================
   Reduction
   ================================================================ */

extern "C" TensorHandle tensor_sum_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::sum(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SUM, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_sum(TensorHandle h) {
    return tensor_sum_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_sum_dim_mlx_streamed(TensorHandle h, int dim, int keepdim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    int rank = (int)t->data.ndim();
    int normalized = dim < 0 ? dim + rank : dim;
    auto r = new Tensor(
        mx::sum(t->data, std::vector<int>{normalized}, keepdim != 0),
        t->requires_grad);
    if (t->requires_grad) {
        int idx = tape_append(OP_SUM_DIM, r, t, nullptr, 0);
        auto meta = new SumDimReplayMeta{normalized, keepdim != 0 ? 1 : 0};
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_sum_dim(TensorHandle h, int dim, int keepdim) {
    return tensor_sum_dim_mlx_streamed(h, dim, keepdim, default_stream_tag());
}
extern "C" TensorHandle tensor_mean_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::mean(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MEAN, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_mean(TensorHandle h) {
    return tensor_mean_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_min_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto result = mx::min(t->data);
    mx::eval(result);
    return (TensorHandle)new Tensor(result, false);

}
TensorHandle tensor_min(TensorHandle h) {
    return tensor_min_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_max_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto result = mx::max(t->data);
    mx::eval(result);
    return (TensorHandle)new Tensor(result, false);

}
TensorHandle tensor_max(TensorHandle h) {
    return tensor_max_mlx_streamed(h, default_stream_tag());
}

/* ================================================================
   Linear algebra
   ================================================================ */

extern "C" TensorHandle tensor_matmul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_matmul(TensorHandle ha, TensorHandle hb) {
    return tensor_matmul_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_mv_mlx_streamed(TensorHandle hmat, TensorHandle hvec, int stream_tag) {
    WITH_STREAM(stream_tag);

    // mat=[m,n], vec=[n] → result=[m]
    auto mat = (Tensor*)hmat; auto vec = (Tensor*)hvec;
    int n = (int)vec->data.size();
    int m_size = (int)mat->data.shape(0);
    auto vec_col = mx::reshape(vec->data, {n, 1});
    auto result_col = mx::matmul(mat->data, vec_col); // [m, 1]
    auto result = mx::reshape(result_col, {m_size});   // [m]
    bool rg = mat->requires_grad || vec->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_MV, r, mat, vec, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
    return tensor_mv_mlx_streamed(hmat, hvec, default_stream_tag());
}

extern "C" TensorHandle tensor_linear_mlx_streamed(TensorHandle hW, TensorHandle hx, TensorHandle hbias, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* Decompose into mv + add so the bias dependency lands on the tape.
       The previous fused form recorded only OP_MV(W,x), dropping the bias
       from the replay graph — when tlinear chained (one tlinear's output
       used as the next tlinear's bias), the inner branch had no path to
       the loss in the VJP and gradients on those params went to zero. */
    TensorHandle mv_h = tensor_mv_mlx_streamed(hW, hx, stream_tag);
    if (!hbias) return mv_h;
    return tensor_add_mlx_streamed(mv_h, hbias, stream_tag);

}
TensorHandle tensor_linear(TensorHandle hW, TensorHandle hx, TensorHandle hbias) {
    return tensor_linear_mlx_streamed(hW, hx, hbias, default_stream_tag());
}

extern "C" TensorHandle tensor_linear_2d_mlx_streamed(TensorHandle hW, TensorHandle hX, TensorHandle hbias, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* W: [o, i], X: [B, i], bias: [o] -> Y: [B, o] = X @ W^T + bias */
    auto W = (Tensor*)hW; auto X = (Tensor*)hX; auto bias = (Tensor*)hbias;
    auto WT = mx::transpose(W->data, {1, 0});
    auto result = mx::matmul(X->data, WT);
    if (bias) result = mx::add(result, bias->data);
    bool rg = W->requires_grad || X->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_LINEAR_2D, r, X, W, 0);
        auto meta = new LinearReplayMeta();
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_linear_2d(TensorHandle hW, TensorHandle hX, TensorHandle hbias) {
    return tensor_linear_2d_mlx_streamed(hW, hX, hbias, default_stream_tag());
}

extern "C" TensorHandle tensor_concat_2d_axis1_mlx_streamed(TensorHandle hA, TensorHandle hB, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* A: [m, n], B: [m, k] -> [m, n+k] along axis 1 */
    auto A = (Tensor*)hA; auto B = (Tensor*)hB;
    auto result = mx::concatenate({A->data, B->data}, 1);
    bool rg = A->requires_grad || B->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_CONCAT_2D_AXIS1, r, A, B, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_concat_2d_axis1(TensorHandle hA, TensorHandle hB) {
    return tensor_concat_2d_axis1_mlx_streamed(hA, hB, default_stream_tag());
}

extern "C" TensorHandle tensor_dot_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::sum(mx::multiply(a->data, b->data)), rg);
    // Use OP_MUL + OP_SUM for backward (approximate)
    if (rg) {
        auto prod = new Tensor(mx::multiply(a->data, b->data), rg);
        tape_append(OP_MUL, prod, a, b, 0);
        tape_append(OP_SUM, r, prod, nullptr, 0);
    }
    return (TensorHandle)r;

}
TensorHandle tensor_dot(TensorHandle ha, TensorHandle hb) {
    return tensor_dot_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_outer_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::outer(a->data, b->data), rg);
    if (rg) tape_append(OP_OUTER, r, a, b, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_outer(TensorHandle ha, TensorHandle hb) {
    return tensor_outer_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_softmax_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, dim), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_softmax(TensorHandle h, int dim) {
    return tensor_softmax_mlx_streamed(h, dim, default_stream_tag());
}
extern "C" TensorHandle tensor_log_softmax_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    // log_softmax = x - log(sum(exp(x)))
    auto maxv = mx::max(t->data, dim, true);
    auto shifted = mx::subtract(t->data, maxv);
    auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
    auto r = new Tensor(mx::subtract(t->data, lse), t->requires_grad);
    if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, (double)dim);
    return (TensorHandle)r;

}
TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
    return tensor_log_softmax_mlx_streamed(h, dim, default_stream_tag());
}

/* ================================================================
   Loss functions
   ================================================================ */

extern "C" TensorHandle tensor_bce_with_logits_mlx_streamed(TensorHandle hinput, TensorHandle htarget, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* BCE with logits = mean(max(x,0) - x*y + log(1 + exp(-|x|))).
       Decomposed into primitive ops so each step records its own tape entry —
       backward flows automatically through replay-based vjp. Without the
       decomposition, the fused result has no tape entry, `tape_idx` stays -1,
       and `tensor_backward` returns early — params never receive gradients. */
    TensorHandle relu_x = tensor_clamp_min_mlx_streamed(hinput, 0.0, stream_tag);
    TensorHandle xy = tensor_mul_mlx_streamed(hinput, htarget, stream_tag);
    TensorHandle abs_x = tensor_abs_mlx_streamed(hinput, stream_tag);
    TensorHandle neg_abs_x = tensor_neg_mlx_streamed(abs_x, stream_tag);
    TensorHandle exp_neg = tensor_exp_mlx_streamed(neg_abs_x, stream_tag);
    TensorHandle one_plus_exp = tensor_add_scalar_mlx_streamed(exp_neg, 1.0, stream_tag);
    TensorHandle log_term = tensor_log_mlx_streamed(one_plus_exp, stream_tag);
    TensorHandle relu_minus_xy = tensor_sub_mlx_streamed(relu_x, xy, stream_tag);
    TensorHandle inner = tensor_add_mlx_streamed(relu_minus_xy, log_term, stream_tag);
    return tensor_mean_mlx_streamed(inner, stream_tag);

}
TensorHandle tensor_bce_with_logits(TensorHandle hinput, TensorHandle htarget) {
    return tensor_bce_with_logits_mlx_streamed(hinput, htarget, default_stream_tag());
}
TensorHandle tensor_cross_entropy(TensorHandle hinput, TensorHandle htarget) {
    /* Cross-entropy with soft labels: CE = -mean(target * log_softmax(input)).
       Decomposed into primitives so each step records its own tape entry —
       backward flows automatically through replay-based vjp.

       Matches tape backend's choice of dim=0 for log_softmax for cross-backend
       consistency. */
    TensorHandle ls = tensor_log_softmax(hinput, 0);
    TensorHandle prod = tensor_mul(htarget, ls);
    TensorHandle neg = tensor_neg(prod);
    return tensor_mean(neg);
}

TensorHandle tensor_mse_loss(TensorHandle hinput, TensorHandle htarget) {
    /* MSE = mean((input - target)^2). Decomposed via existing primitives. */
    TensorHandle diff = tensor_sub(hinput, htarget);
    TensorHandle sq = tensor_mul(diff, diff);
    return tensor_mean(sq);
}

/* ================================================================
   NTM-specific
   ================================================================ */

extern "C" TensorHandle tensor_cosine_similarity_mlx_streamed(TensorHandle hmemory, TensorHandle hkey, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    // memory=[n,m], key=[m] → result=[n]
    auto mem = (Tensor*)hmemory; auto key = (Tensor*)hkey;
    auto eps = scalar_like(1.0e-8, mem->data);
    int n = (int)mem->data.shape(0);
    int m = (int)mem->data.shape(1);

    // Compute forward
    auto key_2d = mx::reshape(key->data, {1, m});
    auto dots = mx::sum(mx::multiply(mem->data, key_2d), std::vector<int>{1}); // [n]
    auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(mem->data), std::vector<int>{1}), eps)); // [n]
    auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(key->data)), eps)); // scalar
    auto result = mx::divide(dots, mx::multiply(row_norms, key_norm));

    bool rg = mem->requires_grad || key->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_COSINE_SIM, r, mem, key, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_cosine_similarity(TensorHandle hmemory, TensorHandle hkey, int dim) {
    return tensor_cosine_similarity_mlx_streamed(hmemory, hkey, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_conv1d_circular_mlx_streamed(TensorHandle hinput, TensorHandle hkernel, int stream_tag) {
    WITH_STREAM(stream_tag);

    // Circular correlation: out[i] = sum_j input[(i-k/2+j+n)%n] * kernel[j]
    auto inp = (Tensor*)hinput; auto kern = (Tensor*)hkernel;
    int n = (int)inp->data.size();
    int k = (int)kern->data.size();

    mx::array result = mx::zeros({n}, inp->data.dtype());
    int half_k = k / 2;
    for (int j = 0; j < k; j++) {
        int shift = half_k - j;
        auto shifted = mx::roll(inp->data, shift);
        auto kern_j = mx::take(kern->data, mx::array(j));
        result = mx::add(result, mx::multiply(shifted, kern_j));
    }

    bool rg = inp->requires_grad || kern->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_CONV1D_CIRC, r, inp, kern, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    return tensor_conv1d_circular_mlx_streamed(hinput, hkernel, default_stream_tag());
}

extern "C" TensorHandle tensor_batch_norm_mlx_streamed(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               TensorHandle hrunning_mean, TensorHandle hrunning_var,
                               int C, int spatial, int training,
                               double momentum, double eps, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;
    auto gamma = (Tensor*)hgamma;
    auto beta = (Tensor*)hbeta;
    auto rm = (Tensor*)hrunning_mean;
    auto rv = (Tensor*)hrunning_var;

    // Reshape flat input to [C, spatial]
    auto x = mx::reshape(inp->data, {C, spatial});
    auto mean = mx::mean(x, std::vector<int>{1}, true);  // [C, 1]
    auto var = mx::var(x, std::vector<int>{1}, true);     // [C, 1]

    if (training) {
        // Update running stats (non-differentiable)
        auto mom      = scalar_like(momentum,       rm->data);
        auto one_m_mo = scalar_like(1.0 - momentum, rm->data);
        auto new_rm = mx::add(mx::multiply(one_m_mo, rm->data),
                              mx::multiply(mom,     mx::squeeze(mean)));
        auto new_rv = mx::add(mx::multiply(one_m_mo, rv->data),
                              mx::multiply(mom,     mx::squeeze(var)));
        rm->data = new_rm;
        rv->data = new_rv;
        mx::eval(rm->data);
        mx::eval(rv->data);
    } else {
        mean = mx::reshape(rm->data, {C, 1});
        var = mx::reshape(rv->data, {C, 1});
    }

    auto rstd = mx::rsqrt(mx::add(var, scalar_like(eps, var)));
    auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
    auto g = mx::reshape(gamma->data, {C, 1});
    auto b = mx::reshape(beta->data, {C, 1});
    auto result = mx::flatten(mx::add(mx::multiply(g, x_hat), b));

    bool rg = inp->requires_grad || gamma->requires_grad || beta->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_BATCH_NORM, r, inp, nullptr, 0);
        auto* meta = new BatchNormReplayMeta();
        meta->gamma_pool_idx = gamma->pool_idx;
        meta->beta_pool_idx = beta->pool_idx;
        meta->C = C;
        meta->spatial = spatial;
        meta->eps = eps;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               TensorHandle hrunning_mean, TensorHandle hrunning_var,
                               int C, int spatial, int training,
                               double momentum, double eps) {
    return tensor_batch_norm_mlx_streamed(hinput, hgamma, hbeta, hrunning_mean, hrunning_var, C, spatial, training, momentum, eps, default_stream_tag());
}

extern "C" TensorHandle tensor_dropout_mlx_streamed(TensorHandle hinput, double p, int training, unsigned int seed, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;
    if (!training || p <= 0.0) return hinput;

    // Generate bernoulli mask and scale by 1/(1-p)
    // MLX random only supports float32 on Metal — generate in f32, compare,
    // build a f32 mask, then cast to the input's dtype so the final multiply
    // doesn't force a dtype promotion on `inp->data`.
    double scale = 1.0 / (1.0 - p);
    auto rnd = mx::random::uniform(kF32_ZERO(), kF32_ONE(), inp->data.shape(), mx::float32);
    auto keep = mx::greater(rnd, mx::array((float)p, mx::float32));
    auto mask_f32 = mx::where(keep, mx::array(scale, mx::float32), kF32_ZERO());
    auto mask = mx::astype(mask_f32, inp->data.dtype());
    auto result = mx::multiply(inp->data, mask);

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        // For replay: store the mask as a constant in the pool so vjp can diff through multiply
        auto mask_t = new Tensor(mask, false);
        int idx = tape_append(OP_DROPOUT, r, inp, mask_t, 0);
    }
    return (TensorHandle)r;

}
TensorHandle tensor_dropout(TensorHandle hinput, double p, int training, unsigned int seed) {
    return tensor_dropout_mlx_streamed(hinput, p, training, seed, default_stream_tag());
}

extern "C" TensorHandle tensor_cross_attention_mlx_streamed(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                    TensorHandle hmask, double scale, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* Compose from existing ops — MLX replay autograd handles backward.
       Thread stream_tag through each inner call so the type-level device
       stays in effect (the unsuffixed sub-op trampolines would each open
       their own WITH_STREAM(default_stream_tag()) and clobber our scope). */
    TensorHandle KT = tensor_transpose_last2_mlx_streamed(hK, stream_tag);
    TensorHandle scores = tensor_mul_scalar_mlx_streamed(
        tensor_bmm_3x3_mlx_streamed(hQ, KT, stream_tag), scale, stream_tag);
    if (hmask) scores = tensor_masked_fill_mlx_streamed(scores, hmask, -1.0e20, stream_tag);
    TensorHandle attn = tensor_softmax_3d_mlx_streamed(scores, stream_tag);
    return tensor_bmm_3x3_mlx_streamed(attn, hV, stream_tag);

}
TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                    TensorHandle hmask, double scale) {
    return tensor_cross_attention_mlx_streamed(hQ, hK, hV, hmask, scale, default_stream_tag());
}

extern "C" TensorHandle tensor_embedding_mlx_streamed(TensorHandle hweight, TensorHandle hindices, int n, int embedDim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto weight = (Tensor*)hweight;
    auto indices = (Tensor*)hindices;
    auto idx_int = mx::astype(indices->data, mx::int32);
    auto rows = mx::take(weight->data, idx_int, 0);  /* [n, embedDim] */
    auto result = mx::flatten(rows);  /* [n * embedDim] */

    auto r = new Tensor(result, weight->requires_grad);
    if (weight->requires_grad) {
        // For replay: store indices as arg2 so vjp can differentiate through take
        auto idx_t = new Tensor(idx_int, false);
        tape_append(OP_EMBEDDING, r, weight, idx_t, (double)embedDim);
    }
    return (TensorHandle)r;

}
TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    return tensor_embedding_mlx_streamed(hweight, hindices, n, embedDim, default_stream_tag());
}

extern "C" TensorHandle tensor_gather_mlx_streamed(TensorHandle hinput, TensorHandle hindex, int n, int stream_tag) {
    WITH_STREAM(stream_tag);

    (void)n;
    auto inp = (Tensor*)hinput;
    auto idx = (Tensor*)hindex;
    auto idx_int = mx::astype(idx->data, mx::int32);
    auto result = mx::take(inp->data, idx_int, 0);
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_GATHER, r, inp, idx, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    return tensor_gather_mlx_streamed(hinput, hindex, n, default_stream_tag());
}

extern "C" TensorHandle tensor_scatter_add_mlx_streamed(TensorHandle hindex, TensorHandle hsrc, int out_size, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto idx = (Tensor*)hindex;
    auto src = (Tensor*)hsrc;
    auto idx_int = mx::astype(idx->data, mx::int32);
    auto base = mx::zeros({out_size}, src->data.dtype());
    /* mx::scatter_add updates shape: indices.shape + base.shape[axis+1:].
       For 1D base on axis 0 that's [N, 1] (the trailing 1 is the empty
       remainder reified as a singleton). */
    auto updates_2d = mx::reshape(src->data, {(int)src->data.size(), 1});
    auto result = mx::scatter_add(base, {idx_int}, updates_2d, std::vector<int>{0});
    auto r = new Tensor(result, src->requires_grad);
    if (src->requires_grad) tape_append(OP_SCATTER_ADD, r, src, idx, (double)out_size);
    return (TensorHandle)r;

}
TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
    return tensor_scatter_add_mlx_streamed(hindex, hsrc, out_size, default_stream_tag());
}

extern "C" TensorHandle tensor_argsort_mlx_streamed(TensorHandle ht, int dim, int descending, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)ht;
    // MLX argsort returns ascending by default
    auto indices = mx::argsort(t->data, dim);
    if (descending) {
        // Reverse: take from end
        int n = (int)t->data.size();
        auto rev_idx = mx::subtract(mx::array(n - 1), mx::arange(n));
        indices = mx::take(indices, rev_idx);
    }
    auto result = mx::astype(indices, mx::float32);
    mx::eval(result);
    return (TensorHandle)(new Tensor(result, false)); // no grad for indices

}
TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
    return tensor_argsort_mlx_streamed(ht, dim, descending, default_stream_tag());
}

extern "C" TensorHandle tensor_cumprod_mlx_streamed(TensorHandle ht, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)ht;
    auto result = mx::cumprod(t->data, dim);
    auto r = new Tensor(result, t->requires_grad);
    if (r->requires_grad) {
        tape_append(OP_CUMPROD, r, t, NULL, 0.0);
    }
    return (TensorHandle)r;

}
TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    return tensor_cumprod_mlx_streamed(ht, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_gru_cell_mlx_streamed(TensorHandle hih, TensorHandle hhh,
                              TensorHandle hprev, int o, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* nn.GRU equation. ih = W_ih @ x + b_ih, hh = W_hh @ h + b_hh.
       MLX replay autograd handles backward via the closure. */
    auto ih = (Tensor*)hih;
    auto hh = (Tensor*)hhh;
    auto prev = (Tensor*)hprev;
    auto ih_z = mx::slice(ih->data, {0}, {o});
    auto ih_r = mx::slice(ih->data, {o}, {2*o});
    auto ih_n = mx::slice(ih->data, {2*o}, {3*o});
    auto hh_z = mx::slice(hh->data, {0}, {o});
    auto hh_r = mx::slice(hh->data, {o}, {2*o});
    auto hh_n = mx::slice(hh->data, {2*o}, {3*o});
    auto z = mx::sigmoid(mx::add(ih_z, hh_z));
    auto r_gate = mx::sigmoid(mx::add(ih_r, hh_r));
    auto n = mx::tanh(mx::add(ih_n, mx::multiply(r_gate, hh_n)));
    auto result = mx::add(mx::multiply(mx::subtract(one_like(z), z), n),
                          mx::multiply(z, prev->data));

    bool rg = ih->requires_grad || hh->requires_grad || prev->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        /* arg1=ih, arg2=hh; prev (3rd input) goes in op_meta. */
        int idx = tape_append(OP_GRU_CELL, r, ih, hh, 0);
        auto meta = new GruCellReplayMeta();
        meta->o = o;
        meta->prev_pool_idx = prev->pool_idx;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh,
                              TensorHandle hprev, int o) {
    return tensor_gru_cell_mlx_streamed(hih, hhh, hprev, o, default_stream_tag());
}

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    /* Same loop as tape backend — MLX doesn't have native group_norm */
    auto inp = (Tensor*)hinput;
    auto gamma = (Tensor*)hgamma;
    auto beta = (Tensor*)hbeta;
    int n = channels * spatial;
    int chPerGroup = channels / numGroups;
    int groupSize = chPerGroup * spatial;
    mx::eval(inp->data); mx::eval(gamma->data); mx::eval(beta->data);
    /* Stage inputs as double regardless of underlying dtype, so the CPU
       loop is dtype-agnostic. */
    std::vector<double> inpD_buf((size_t)n);
    std::vector<double> gammaD_buf((size_t)channels);
    std::vector<double> betaD_buf((size_t)channels);
    mx_to_doubles(inp->data, inpD_buf.data());
    mx_to_doubles(gamma->data, gammaD_buf.data());
    mx_to_doubles(beta->data, betaD_buf.data());
    const double* inpD = inpD_buf.data();
    const double* gammaD = gammaD_buf.data();
    const double* betaD = betaD_buf.data();
    double* out = (double*)calloc(n, sizeof(double));
    for (int g = 0; g < numGroups; g++) {
        double mean = 0;
        int base = g * groupSize;
        for (int j = 0; j < groupSize; j++) mean += inpD[base + j];
        mean /= groupSize;
        double var = 0;
        for (int j = 0; j < groupSize; j++) { double d = inpD[base+j] - mean; var += d*d; }
        var /= groupSize;
        double rstd = 1.0 / sqrt(var + eps);
        for (int c = 0; c < chPerGroup; c++) {
            int absC = g * chPerGroup + c;
            for (int s = 0; s < spatial; s++) {
                int idx = absC * spatial + s;
                double x_hat = (inpD[idx] - mean) * rstd;
                out[idx] = gammaD[absC] * x_hat + betaD[absC];
            }
        }
    }
    auto result = mx_array_from_doubles(out, {n}, inp->data.dtype());
    free(out);
    return (TensorHandle)(new Tensor(result, inp->requires_grad || gamma->requires_grad));
}

TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int pad, int stride) {
    /* Implement as naive loop (same as tape) since MLX doesn't expose conv_transpose1d directly */
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int outC = (int)ker->data.shape(1), kL = (int)ker->data.shape(2);
    int oL = (L - 1) * stride - 2 * pad + kL;

    // Compute on CPU via eval then manual scatter
    mx::eval(inp->data); mx::eval(ker->data);
    std::vector<double> inpD_buf((size_t)(inC * L));
    std::vector<double> kerD_buf((size_t)(inC * outC * kL));
    mx_to_doubles(inp->data, inpD_buf.data());
    mx_to_doubles(ker->data, kerD_buf.data());
    const double* inpD = inpD_buf.data();
    const double* kerD = kerD_buf.data();
    double* out = (double*)calloc(outC * oL, sizeof(double));
    if (bias) {
        mx::eval(bias->data);
        std::vector<double> biasD_buf((size_t)outC);
        mx_to_doubles(bias->data, biasD_buf.data());
        const double* biasD = biasD_buf.data();
        for (int oc = 0; oc < outC; oc++) for (int ol = 0; ol < oL; ol++) out[oc*oL+ol] = biasD[oc];
    }
    for (int ic = 0; ic < inC; ic++)
        for (int il = 0; il < L; il++)
            for (int oc = 0; oc < outC; oc++)
                for (int kl = 0; kl < kL; kl++) {
                    int ol = il*stride - pad + kl;
                    if (ol >= 0 && ol < oL)
                        out[oc*oL+ol] += inpD[ic*L+il] * kerD[ic*outC*kL+oc*kL+kl];
                }
    auto result = mx_array_from_doubles(out, {outC, oL}, inp->data.dtype());
    free(out);
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int padH, int padW,
                                     int strideH, int strideW) {
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int outC = (int)ker->data.shape(1), kH = (int)ker->data.shape(2), kW = (int)ker->data.shape(3);
    int oH = (H-1)*strideH - 2*padH + kH;
    int oW = (W-1)*strideW - 2*padW + kW;
    mx::eval(inp->data); mx::eval(ker->data);
    std::vector<double> inpD_buf((size_t)(inC * H * W));
    std::vector<double> kerD_buf((size_t)(inC * outC * kH * kW));
    mx_to_doubles(inp->data, inpD_buf.data());
    mx_to_doubles(ker->data, kerD_buf.data());
    const double* inpD = inpD_buf.data();
    const double* kerD = kerD_buf.data();
    double* out = (double*)calloc(outC*oH*oW, sizeof(double));
    if (bias) {
        mx::eval(bias->data);
        std::vector<double> biasD_buf((size_t)outC);
        mx_to_doubles(bias->data, biasD_buf.data());
        const double* biasD = biasD_buf.data();
        for (int oc = 0; oc < outC; oc++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) out[oc*oH*oW+oh*oW+ow] = biasD[oc];
    }
    for (int ic = 0; ic < inC; ic++)
        for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
                for (int oc = 0; oc < outC; oc++)
                    for (int kh = 0; kh < kH; kh++)
                        for (int kw = 0; kw < kW; kw++) {
                            int oh = ih*strideH - padH + kh;
                            int ow = iw*strideW - padW + kw;
                            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
                                out[oc*oH*oW+oh*oW+ow] += inpD[ic*H*W+ih*W+iw]
                                    * kerD[ic*outC*kH*kW+oc*kH*kW+kh*kW+kw];
                        }
    auto result = mx_array_from_doubles(out, {outC, oH, oW}, inp->data.dtype());
    free(out);
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int pad, int stride, int groups) {
    if (groups == 1) return tensor_conv1d(hinput, hkernel, hbias, pad, stride);
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    auto inp_lc = mx::transpose(inp->data, {1, 0});
    auto inp_nlc = mx::reshape(inp_lc, {1, L, inC});
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 1});
    auto out = mx::conv1d(inp_nlc, ker_mlx, stride, pad, /*dilation=*/1, groups);
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {1, 0});
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1}));
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int padH, int padW,
                                   int strideH, int strideW, int groups) {
    if (groups == 1) return tensor_conv2d(hinput, hkernel, hbias, padH, padW, strideH, strideW);
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    auto inp_hwc = mx::transpose(inp->data, {1, 2, 0});
    auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC});
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});
    auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW}, /*dilation=*/{1, 1}, groups);
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {2, 0, 1});
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

extern "C" TensorHandle tensor_avg_pool1d_mlx_streamed(TensorHandle hinput, int kL, int stride, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;
    auto dt = inp->data.dtype();
    int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int oL = (L - kL) / stride + 1;
    // Sum via strided slices, then divide by kL
    mx::array result = mx::zeros({C, oL}, dt);
    for (int kl = 0; kl < kL; kl++) {
        auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
        result = mx::add(result, sliced);
    }
    result = mx::divide(result, mx::array((double)kL, dt));
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_AVG_POOL1D, r, inp, nullptr, (double)kL + stride * 0.001);
    return (TensorHandle)r;

}
TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
    return tensor_avg_pool1d_mlx_streamed(hinput, kL, stride, default_stream_tag());
}

extern "C" TensorHandle tensor_avg_pool2d_mlx_streamed(TensorHandle hinput, int kH, int kW, int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;
    auto dt = inp->data.dtype();
    int C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    mx::array result = mx::zeros({C, oH, oW}, dt);
    for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, kh, kw}, {C, kh + oH * strideH, kw + oW * strideW}, {1, strideH, strideW});
            result = mx::add(result, sliced);
        }
    result = mx::divide(result, mx::array((double)(kH * kW), dt));
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_AVG_POOL2D, r, inp, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    return tensor_avg_pool2d_mlx_streamed(hinput, kH, kW, strideH, strideW, default_stream_tag());
}

extern "C" TensorHandle tensor_conv1d_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int pad, int stride, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);

    // MLX conv1d: input [N, L, C_in], weight [C_out, kL, C_in]
    auto inp_lc = mx::transpose(inp->data, {1, 0});  // [L, inC]
    auto inp_nlc = mx::reshape(inp_lc, {1, L, inC});
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 1});  // [outC, kL, inC]
    auto out = mx::conv1d(inp_nlc, ker_mlx, stride, pad);
    auto out_sq = mx::squeeze(out, 0);  // [oL, outC]
    auto result = mx::transpose(out_sq, {1, 0});  // [outC, oL]
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV1D, r, inp, ker, 0);
        auto* meta = new Conv1DReplayMeta();
        meta->pad = pad; meta->stride = stride; meta->inC = inC; meta->L = L;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int pad, int stride) {
    return tensor_conv1d_mlx_streamed(hinput, hkernel, hbias, pad, stride, default_stream_tag());
}

extern "C" TensorHandle tensor_max_pool1d_mlx_streamed(TensorHandle hinput, int kL, int stride, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;
    int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int oL = (L - kL) / stride + 1;

    mx::array result = mx::full({C, oL}, -1e30, inp->data.dtype());
    for (int kl = 0; kl < kL; kl++) {
        auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
        result = mx::maximum(result, sliced);
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL1D, r, inp, nullptr, 0);
        auto* meta = new MaxPool1DReplayMeta();
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    return tensor_max_pool1d_mlx_streamed(hinput, kL, stride, default_stream_tag());
}

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

extern "C" TensorHandle tensor_conv2d_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;   // [inC, H, W]
    auto ker = (Tensor*)hkernel;  // [outC, inC, kH, kW]
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;

    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);

    // MLX conv2d expects NHWC: input [N,H,W,C_in], weight [C_out,kH,kW,C_in]
    auto inp_hwc = mx::transpose(inp->data, {1, 2, 0});  // [H, W, inC]
    auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC}); // [1, H, W, inC]
    // kernel: [outC, inC, kH, kW] -> [outC, kH, kW, inC]
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});

    auto out = mx::conv2d(inp_nhwc, ker_mlx,
                          {strideH, strideW}, {padH, padW});
    // out: [1, oH, oW, outC] -> squeeze batch -> [oH, oW, outC] -> [outC, oH, oW]
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {2, 0, 1});

    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV2D, r, inp, ker, 0);
        auto* meta = new Conv2DReplayMeta();
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->inC = inC; meta->H = H; meta->W = W;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW) {
    return tensor_conv2d_mlx_streamed(hinput, hkernel, hbias, padH, padW, strideH, strideW, default_stream_tag());
}

extern "C" TensorHandle tensor_max_pool2d_mlx_streamed(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;  // [C, H, W]
    int C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;

    // Max pool via strided slicing: for each (kh,kw) offset, slice with stride and take max
    mx::array result = mx::full({C, oH, oW}, -1e30, inp->data.dtype());
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, kh, kw},
                {C, kh + oH * strideH, kw + oW * strideW},
                {1, strideH, strideW});
            result = mx::maximum(result, sliced);
        }
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL2D, r, inp, nullptr, 0);
        auto* meta = new MaxPool2DReplayMeta();
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW) {
    return tensor_max_pool2d_mlx_streamed(hinput, kH, kW, strideH, strideW, default_stream_tag());
}

extern "C" TensorHandle tensor_conv2d_batched_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                                    TensorHandle hbias, int padH, int padW,
                                    int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;   // [B, inC, H, W]
    auto ker = (Tensor*)hkernel;  // [outC, inC, kH, kW]
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;

    int B = (int)inp->data.shape(0), inC = (int)inp->data.shape(1);
    int H = (int)inp->data.shape(2), W = (int)inp->data.shape(3);

    // MLX conv2d expects NHWC: [B, H, W, inC]; kernel [outC, kH, kW, inC]
    auto inp_nhwc = mx::transpose(inp->data, {0, 2, 3, 1});
    auto ker_mlx  = mx::transpose(ker->data, {0, 2, 3, 1});

    auto out = mx::conv2d(inp_nhwc, ker_mlx,
                          {strideH, strideW}, {padH, padW});
    // out: [B, oH, oW, outC] -> [B, outC, oH, oW]
    auto result = mx::transpose(out, {0, 3, 1, 2});

    if (bias) result = mx::add(result, mx::reshape(bias->data, {1, -1, 1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV2D_BATCHED, r, inp, ker, 0);
        auto* meta = new Conv2DBatchedReplayMeta();
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->B = B; meta->inC = inC; meta->H = H; meta->W = W;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                    TensorHandle hbias, int padH, int padW,
                                    int strideH, int strideW) {
    return tensor_conv2d_batched_mlx_streamed(hinput, hkernel, hbias, padH, padW, strideH, strideW, default_stream_tag());
}

extern "C" TensorHandle tensor_max_pool2d_batched_mlx_streamed(TensorHandle hinput, int kH, int kW,
                                        int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto inp = (Tensor*)hinput;  // [B, C, H, W]
    int B = (int)inp->data.shape(0), C = (int)inp->data.shape(1);
    int H = (int)inp->data.shape(2), W = (int)inp->data.shape(3);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;

    mx::array result = mx::full({B, C, oH, oW}, -1e30, inp->data.dtype());
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, 0, kh, kw},
                {B, C, kh + oH * strideH, kw + oW * strideW},
                {1, 1, strideH, strideW});
            result = mx::maximum(result, sliced);
        }
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL2D_BATCHED, r, inp, nullptr, 0);
        auto* meta = new MaxPool2DBatchedReplayMeta();
        meta->B = B; meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW,
                                        int strideH, int strideW) {
    return tensor_max_pool2d_batched_mlx_streamed(hinput, kH, kW, strideH, strideW, default_stream_tag());
}

/* ================================================================
   Shape manipulation
   ================================================================ */

extern "C" TensorHandle tensor_reshape_mlx_streamed(TensorHandle h, int* shape, int rank, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    mx::Shape sh(shape, shape + rank);
    auto r = new Tensor(mx::reshape(t->data, sh), t->requires_grad);
    if (t->requires_grad) tape_append(OP_RESHAPE, r, t, nullptr, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    return tensor_reshape_mlx_streamed(h, shape, rank, default_stream_tag());
}

extern "C" TensorHandle tensor_unsqueeze_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    const auto& orig = t->data.shape();
    int rank = (int)orig.size();
    std::vector<int> new_dims;
    new_dims.reserve(rank + 1);
    for (int i = 0; i <= rank; i++) {
        if (i == dim) new_dims.push_back(1);
        if (i < rank) new_dims.push_back(orig[i]);
    }
    mx::Shape sh(new_dims.begin(), new_dims.end());
    auto r = new Tensor(mx::reshape(t->data, sh), t->requires_grad);
    if (t->requires_grad) tape_append(OP_RESHAPE, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
    return tensor_unsqueeze_mlx_streamed(h, dim, default_stream_tag());
}
extern "C" TensorHandle tensor_squeeze_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    int rank = (int)t->data.ndim();
    int normalized = dim < 0 ? dim + rank : dim;
    /* No-op if dim is out of range or not size 1 — matches torch's .squeeze(dim) */
    if (normalized < 0 || normalized >= rank || (int)t->data.shape(normalized) != 1) {
        return tensor_clone_mlx_streamed(h, stream_tag);
    }
    std::vector<int> new_shape;
    new_shape.reserve(rank - 1);
    for (int i = 0; i < rank; i++) {
        if (i != normalized) new_shape.push_back((int)t->data.shape(i));
    }
    /* Reshape preserves data layout: squeeze of a size-1 dim is identity on data.
       Reuse OP_RESHAPE so backward replay reconstructs the same shape. */
    return tensor_reshape_mlx_streamed(h, new_shape.data(), (int)new_shape.size(), stream_tag);

}
TensorHandle tensor_squeeze(TensorHandle h, int dim) {
    return tensor_squeeze_mlx_streamed(h, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_select_mlx_streamed(TensorHandle h, int dim, int index, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::take(t->data, mx::array(index), dim), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SELECT, r, t, nullptr, (double)index);
    return (TensorHandle)r;

}
TensorHandle tensor_select(TensorHandle h, int dim, int index) {
    return tensor_select_mlx_streamed(h, dim, index, default_stream_tag());
}

extern "C" TensorHandle tensor_stack_mlx_streamed(TensorHandle* tensors, int count, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* Same shape as tensor_stack_from_array, but the caller (test_backend or
       internal C code) retains ownership of the input handle array — we do
       NOT free it. tensor_stack_from_array is the variant that takes
       ownership of an Idris-allocated handle array. */
    std::vector<mx::array> arrs;
    bool rg = false;
    for (int i = 0; i < count; i++) {
        auto t = (Tensor*)tensors[i];
        arrs.push_back(t->data);
        if (t->requires_grad) rg = true;
    }
    auto r = new Tensor(mx::stack(arrs, dim), rg);
    if (rg) {
        int idx = tape_append(OP_STACK, r, nullptr, nullptr, (double)dim);
        auto* indices = new std::vector<int>();
        for (int i = 0; i < count; i++)
            indices->push_back(((Tensor*)tensors[i])->pool_idx);
        tape[idx].meta = (void*)indices;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) {
    return tensor_stack_mlx_streamed(tensors, count, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_cat_mlx_streamed(TensorHandle* tensors, int count, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);

    std::vector<mx::array> arrs;
    bool rg = false;
    for (int i = 0; i < count; i++) {
        auto t = (Tensor*)tensors[i];
        arrs.push_back(t->data);
        if (t->requires_grad) rg = true;
    }
    auto r = new Tensor(mx::concatenate(arrs, dim), rg);
    if (rg) {
        int idx = tape_append(OP_CAT_MULTI, r, nullptr, nullptr, (double)dim);
        auto* indices = new std::vector<int>();
        for (int i = 0; i < count; i++)
            indices->push_back(((Tensor*)tensors[i])->pool_idx);
        tape[idx].meta = (void*)indices;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) {
    return tensor_cat_mlx_streamed(tensors, count, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_cat2_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::concatenate({a->data, b->data}, 0), rg);
    if (rg) tape_append(OP_CAT, r, a, b, (double)a->data.size());
    return (TensorHandle)r;

}
TensorHandle tensor_cat2(TensorHandle ha, TensorHandle hb) {
    return tensor_cat2_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_narrow_mlx_streamed(TensorHandle h, int dim, int start, int len, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    // Flatten, then slice the 1D range — matches tape backend semantics
    auto flat = mx::flatten(t->data);
    auto sliced = mx::slice(flat, mx::Shape{start}, mx::Shape{start + len});
    auto r = new Tensor(sliced, t->requires_grad);
    if (t->requires_grad) tape_append(OP_NARROW, r, t, nullptr, (double)start);
    return (TensorHandle)r;

}
TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    return tensor_narrow_mlx_streamed(h, dim, start, len, default_stream_tag());
}

extern "C" TensorHandle tensor_mm_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_mm(TensorHandle ha, TensorHandle hb) {
    return tensor_mm_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_bmm_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_BMM, r, a, b, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_bmm(TensorHandle ha, TensorHandle hb) {
    return tensor_bmm_mlx_streamed(ha, hb, default_stream_tag());
}
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

extern "C" TensorHandle tensor_bmm_3x3_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_BMM_3X3, r, a, b, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_bmm_3x3(TensorHandle ha, TensorHandle hb) {
    return tensor_bmm_3x3_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_softmax_3d_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, -1), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_3D, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_softmax_3d(TensorHandle h) {
    return tensor_softmax_3d_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_transpose_last2_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::transpose(t->data, {0, 2, 1}), t->requires_grad);
    if (t->requires_grad) tape_append(OP_TRANSPOSE_LAST2, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_transpose_last2(TensorHandle h) {
    return tensor_transpose_last2_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_3d_mlx_streamed(TensorHandle h, int d0, int d1, int d2, int stream_tag) {
    WITH_STREAM(stream_tag);

    int shape[] = {d0, d1, d2};
    return tensor_reshape_mlx_streamed(h, shape, 3, stream_tag);

}
TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
    return tensor_reshape_3d_mlx_streamed(h, d0, d1, d2, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_4d_mlx_streamed(TensorHandle h, int d0, int d1, int d2, int d3, int stream_tag) {
    WITH_STREAM(stream_tag);

    int shape[] = {d0, d1, d2, d3};
    return tensor_reshape_mlx_streamed(h, shape, 4, stream_tag);

}
TensorHandle tensor_reshape_4d(TensorHandle h, int d0, int d1, int d2, int d3) {
    return tensor_reshape_4d_mlx_streamed(h, d0, d1, d2, d3, default_stream_tag());
}

extern "C" TensorHandle tensor_expand_mask_mlx_streamed(TensorHandle hmask, int B, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto mask = (Tensor*)hmask;
    int m = mask->data.shape(0), n = mask->data.shape(1);
    // [m,n] → [1,m,n] → broadcast to [B,m,n]
    auto expanded = mx::broadcast_to(mx::reshape(mask->data, {1, m, n}), {B, m, n});
    auto r = new Tensor(expanded, false);
    return (TensorHandle)r;

}
TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
    return tensor_expand_mask_mlx_streamed(hmask, B, default_stream_tag());
}

extern "C" TensorHandle tensor_tile_2d_mlx_streamed(TensorHandle h, int rep0, int rep1, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto tiled = mx::tile(t->data, {rep0, rep1});
    /* When the input is non-grad (e.g. cached positional encoding), the
       tile is a constant from the autograd POV — eagerly materialize so
       mx::vjp sees a leaf and doesn't trace back through the tile op
       (which adds latency on each backward call, costing 10-15% wall on
       small-model shapes). For grad inputs, leave lazy so the tape
       replay can reconstruct the graph. */
    if (!t->requires_grad) {
        mx::eval(tiled);
    }
    auto r = new Tensor(tiled, t->requires_grad);
    if (t->requires_grad) {
        int* meta = (int*)std::malloc(sizeof(int) * 2);
        meta[0] = rep0; meta[1] = rep1;
        int idx = tape_append(OP_TILE_2D, r, t, nullptr, 0);
        if (idx >= 0) tape[idx].meta = meta; else std::free(meta);
    }
    return (TensorHandle)r;
}
TensorHandle tensor_tile_2d(TensorHandle h, int rep0, int rep1) {
    return tensor_tile_2d_mlx_streamed(h, rep0, rep1, default_stream_tag());
}

extern "C" TensorHandle tensor_transpose_2d_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::transpose(t->data, {1, 0}), t->requires_grad);
    if (t->requires_grad) tape_append(OP_TRANSPOSE_2D, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_transpose_2d(TensorHandle h) {
    return tensor_transpose_2d_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_softmax_2d_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, -1), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_softmax_2d(TensorHandle h) {
    return tensor_softmax_2d_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_masked_fill_mlx_streamed(TensorHandle h, TensorHandle hmask, double value, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h; auto mask = (Tensor*)hmask;
    auto val_arr = mx::full(t->data.shape(), value, t->data.dtype());
    auto r = new Tensor(mx::where(mask->data, val_arr, t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MASKED_FILL, r, t, mask, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle hmask, double value) {
    return tensor_masked_fill_mlx_streamed(h, hmask, value, default_stream_tag());
}

extern "C" TensorHandle tensor_log_softmax_2d_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    // log_softmax(x) = x - log(sum(exp(x)))
    auto maxv = mx::max(t->data, -1, true);
    auto shifted = mx::subtract(t->data, maxv);
    auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), -1, true)), maxv);
    auto r = new Tensor(mx::subtract(t->data, lse), t->requires_grad);
    if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, -1.0);
    return (TensorHandle)r;

}
TensorHandle tensor_log_softmax_2d(TensorHandle h) {
    return tensor_log_softmax_2d_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_layer_norm_2d_mlx_streamed(TensorHandle h, TensorHandle hgamma,
    TensorHandle hbias, double eps, int stream_tag) {
    WITH_STREAM(stream_tag);

    auto t = (Tensor*)h;
    auto gamma = (Tensor*)hgamma;
    auto bias = (Tensor*)hbias;
    int m = t->data.shape(0), n = t->data.shape(1);

    auto mean = mx::mean(t->data, -1, true);
    auto centered = mx::subtract(t->data, mean);
    auto var = mx::mean(mx::square(centered), -1, true);
    auto rstd = mx::rsqrt(mx::add(var, scalar_like(eps, var)));
    auto x_hat = mx::multiply(centered, rstd);
    auto result = mx::add(mx::multiply(gamma->data, x_hat), bias->data);

    bool rg = t->requires_grad || gamma->requires_grad || bias->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_LAYER_NORM_2D, r, t, nullptr, eps);
        auto meta = new LayerNormReplayMeta();
        meta->gamma_pool_idx = gamma->pool_idx;
        meta->bias_pool_idx = bias->pool_idx;
        meta->eps = eps;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;

}
TensorHandle tensor_layer_norm_2d(TensorHandle h, TensorHandle hgamma,
    TensorHandle hbias, double eps) {
    return tensor_layer_norm_2d_mlx_streamed(h, hgamma, hbias, eps, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_2d_mlx_streamed(TensorHandle h, int rows, int cols, int stream_tag) {
    WITH_STREAM(stream_tag);

    int shape[] = {rows, cols};
    return tensor_reshape_mlx_streamed(h, shape, 2, stream_tag);

}
TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
    return tensor_reshape_2d_mlx_streamed(h, rows, cols, default_stream_tag());
}

extern "C" TensorHandle tensor_reshape_1d_mlx_streamed(TensorHandle h, int n, int stream_tag) {
    WITH_STREAM(stream_tag);

    int shape[] = {n};
    return tensor_reshape_mlx_streamed(h, shape, 1, stream_tag);

}
TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
    return tensor_reshape_1d_mlx_streamed(h, n, default_stream_tag());
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
static int g_compile_invocations = 0;

void tensor_backward(TensorHandle h) {
    double t0_bwd = _wall_ms_mlx();
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) { prof_backward_ms_mlx += _wall_ms_mlx() - t0_bwd; return; }

    // Collect param pool indices and arrays
    std::vector<int> param_pool_indices;
    std::vector<mx::array> param_arrays;
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (Tensor*)param_tensor(i_);
        param_pool_indices.push_back(tensor->pool_idx);
        param_arrays.push_back(tensor->data);
    }
    if (param_arrays.empty()) return;

    // Build constant pool from tape (O(tape_size), not O(all_tensors)).
    // Index/mask args (OP_GATHER.arg2, OP_SCATTER_ADD.arg2) are discrete and
    // have no derivative — keep them out of the vjp inputs entirely.
    // Replay reads them via closure-captured `e.argN->data` (see below).
    std::vector<std::pair<int, mx::array>> constants;
    std::unordered_set<int> seen;
    for (auto& idx : param_pool_indices) seen.insert(idx);
    auto add_const = [&](Tensor* t) {
        if (t && !seen.count(t->pool_idx)) {
            seen.insert(t->pool_idx);
            constants.emplace_back(t->pool_idx, t->data);
        }
    };
    auto arg2_is_index = [](int op) {
        return op == OP_GATHER || op == OP_SCATTER_ADD;
    };
    for (int i = 0; i <= loss->tape_idx; i++) {
        auto& e = tape[i];
        add_const(e.result);
        add_const(e.arg1);
        if (!arg2_is_index(e.op)) add_const(e.arg2);
    }

    // Capture tape state for the closure
    int loss_pool_idx = loss->pool_idx;
    int loss_tape_idx = loss->tape_idx;
    auto tape_ref = &tape;

    // Job 3 Phase B — explicit-inputs forward. The closure takes
    // [params..., constants...] so that mx::compile (if enabled) does
    // NOT bake per-batch constant values into the compiled graph at
    // trace time. The eager path uses the same closure to keep both
    // paths in lockstep; vjp returns grads for all inputs, but only
    // the leading n_params are written back to param tensors.
    int n_params = (int)param_arrays.size();
    int n_consts = (int)constants.size();
    std::vector<int> constants_pool_indices;
    constants_pool_indices.reserve(n_consts);
    for (auto& [idx, arr] : constants) constants_pool_indices.push_back(idx);

    // Replay forward pass inside mlx::vjp
    int pool_size = next_pool_idx;
    auto forward_fn = [&](const std::vector<mx::array>& xs) -> mx::array {
        // xs[0..n_params) = params, xs[n_params..n_params+n_consts) = constants
        std::vector<mx::array> pool(pool_size, kF32_ZERO());
        for (int i = 0; i < n_params; i++)
            pool[param_pool_indices[i]] = xs[i];
        for (int i = 0; i < n_consts; i++)
            pool[constants_pool_indices[i]] = xs[n_params + i];

        for (int i = 0; i <= loss_tape_idx; i++) {
            auto& e = (*tape_ref)[i];
            int out = e.result->pool_idx;
            auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
            auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();

            switch (e.op) {
            case OP_CONST: break;
            case OP_ADD: pool[out] = mx::add(a, b); break;
            case OP_SUB: pool[out] = mx::subtract(a, b); break;
            case OP_MUL: pool[out] = mx::multiply(a, b); break;
            case OP_DIV: pool[out] = mx::divide(a, b); break;
            case OP_NEG: pool[out] = mx::negative(a); break;
            case OP_ABS: pool[out] = mx::abs(a); break;
            case OP_EXP: pool[out] = mx::exp(a); break;
            case OP_LOG: pool[out] = mx::log(a); break;
            case OP_SQRT: pool[out] = mx::sqrt(a); break;
            case OP_POW: pool[out] = mx::power(a, b); break;
            case OP_SIGMOID: pool[out] = mx::sigmoid(a); break;
            case OP_TANH: pool[out] = mx::tanh(a); break;
            case OP_GELU: {
                auto kGeluC  = scalar_like(0.7978845608028654, a);
                auto kGeluC3 = scalar_like(0.044715,           a);
                auto kThree  = scalar_like(3.0,                a);
                auto inner = mx::multiply(kGeluC, mx::add(a, mx::multiply(kGeluC3, mx::power(a, kThree))));
                pool[out] = mx::multiply(mx::multiply(half_like(a), a), mx::add(one_like(a), mx::tanh(inner)));
                break;
            }
            case OP_LEAKY_RELU: {
                auto alpha = scalar_like(e.scalar_arg, a);
                pool[out] = mx::maximum(mx::multiply(alpha, a), a);
                break;
            }
            case OP_SILU: pool[out] = mx::multiply(a, mx::sigmoid(a)); break;
            case OP_SOFTPLUS: {
                pool[out] = mx::add(mx::maximum(a, zero_like(a)),
                                    mx::log(mx::add(one_like(a), mx::exp(mx::negative(mx::abs(a))))));
                break;
            }
            case OP_TILE_2D: {
                int* reps = (int*)e.meta;
                pool[out] = mx::tile(a, {reps[0], reps[1]});
                break;
            }
            case OP_CAST_DTYPE: {
                mx::Dtype target = (e.scalar_arg == 0.0 ? mx::float32 : mx::float64);
                pool[out] = mx::astype(a, target);
                break;
            }
            case OP_ADD_SCALAR: pool[out] = mx::add(a, scalar_like(e.scalar_arg, a)); break;
            case OP_MUL_SCALAR: pool[out] = mx::multiply(a, scalar_like(e.scalar_arg, a)); break;
            case OP_CLAMP_MIN: pool[out] = mx::maximum(a, scalar_like(e.scalar_arg, a)); break;
            case OP_SUM: pool[out] = mx::sum(a); break;
            case OP_MEAN: pool[out] = mx::mean(a); break;
            case OP_SUM_DIM: {
                auto* sm = (SumDimReplayMeta*)e.meta;
                pool[out] = mx::sum(a, std::vector<int>{sm->dim}, sm->keepdim != 0);
                break;
            }
            case OP_MM: case OP_BMM: case OP_BMM_3X3: pool[out] = mx::matmul(a, b); break;
            case OP_SOFTMAX_3D: pool[out] = mx::softmax(a, -1); break;
            case OP_TRANSPOSE_LAST2: pool[out] = mx::transpose(a, {0, 2, 1}); break;
            case OP_MV: {
                auto col = mx::reshape(b, {(int)b.size(), 1});
                pool[out] = mx::reshape(mx::matmul(a, col), {(int)a.shape(0)});
                break;
            }
            case OP_OUTER: pool[out] = mx::outer(a, b); break;
            case OP_TRANSPOSE_2D: pool[out] = mx::transpose(a, {1, 0}); break;
            case OP_SOFTMAX_2D: pool[out] = mx::softmax(a, -1); break;
            case OP_LOG_SOFTMAX_2D: {
                int dim = (int)e.scalar_arg;  // stored by forward (0 for 1D, -1 for 2D)
                auto maxv = mx::max(a, dim, true);
                auto shifted = mx::subtract(a, maxv);
                auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
                pool[out] = mx::subtract(a, lse);
                break;
            }
            case OP_MASKED_FILL: {
                /* mask `b` may be bool; the fill value must match `a`'s
                   dtype so `mx::where` doesn't force a promotion. */
                auto kNegInfMask = scalar_like(-1e9, a);
                pool[out] = mx::where(b, kNegInfMask, a);
                break;
            }
            case OP_RESHAPE: pool[out] = mx::reshape(a, e.result->data.shape()); break;
            case OP_SELECT: pool[out] = mx::take(a, mx::array((int)e.scalar_arg), 0); break;
            case OP_NARROW: {
                int start = (int)e.scalar_arg;
                int len = (int)e.result->data.size();
                pool[out] = mx::slice(mx::flatten(a), {start}, {start + len});
                break;
            }
            case OP_CAT: pool[out] = mx::concatenate({a, b}, 0); break;
            case OP_STACK: {
                auto* indices = (std::vector<int>*)e.meta;
                if (indices) {
                    std::vector<mx::array> arrs;
                    for (int idx : *indices) arrs.push_back(pool[idx]);
                    pool[out] = mx::stack(arrs, (int)e.scalar_arg);
                }
                break;
            }
            case OP_CAT_MULTI: {
                auto* indices = (std::vector<int>*)e.meta;
                if (indices) {
                    std::vector<mx::array> arrs;
                    for (int idx : *indices) arrs.push_back(pool[idx]);
                    pool[out] = mx::concatenate(arrs, (int)e.scalar_arg);
                }
                break;
            }
            case OP_COSINE_SIM: {
                // Inline cosine similarity forward
                int n = (int)a.shape(0), m = (int)a.shape(1);
                auto key_2d = mx::reshape(b, {1, m});
                auto dots = mx::sum(mx::multiply(a, key_2d), std::vector<int>{1});
                auto eps = scalar_like(1.0e-8, a);
                auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(a), std::vector<int>{1}), eps));
                auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(b)), eps));
                pool[out] = mx::divide(dots, mx::multiply(row_norms, key_norm));
                break;
            }
            case OP_CONV1D_CIRC: {
                // Inline circular convolution forward
                int n = (int)a.size(), k = (int)b.size();
                int half_k = k / 2;
                auto result = mx::zeros({n}, a.dtype());
                for (int j = 0; j < k; j++) {
                    auto shifted = mx::roll(a, half_k - j);
                    auto kern_j = mx::take(b, mx::array(j));
                    result = mx::add(result, mx::multiply(shifted, kern_j));
                }
                pool[out] = result;
                break;
            }
            case OP_LAYER_NORM_2D: {
                auto meta = (LayerNormReplayMeta*)e.meta;
                auto gamma = pool[meta->gamma_pool_idx];
                auto bias = pool[meta->bias_pool_idx];
                auto mean = mx::mean(a, -1, true);
                auto centered = mx::subtract(a, mean);
                auto var = mx::mean(mx::square(centered), -1, true);
                auto rstd = mx::rsqrt(mx::add(var, scalar_like(meta->eps, var)));
                auto x_hat = mx::multiply(centered, rstd);
                pool[out] = mx::add(mx::multiply(gamma, x_hat), bias);
                break;
            }
            case OP_LINEAR_2D: {
                /* a = X [B,i], b = W [o,i]. Y = X @ W^T + bias */
                auto meta = (LinearReplayMeta*)e.meta;
                auto WT = mx::transpose(b, {1, 0});
                auto y = mx::matmul(a, WT);
                if (meta && meta->bias_pool_idx >= 0)
                    y = mx::add(y, pool[meta->bias_pool_idx]);
                pool[out] = y;
                break;
            }
            case OP_CONCAT_2D_AXIS1: {
                /* a = A [m,n], b = B [m,k]. Result = concat along axis 1 -> [m,n+k] */
                pool[out] = mx::concatenate({a, b}, 1);
                break;
            }
            case OP_GRU_CELL: {
                /* nn.GRU: a=ih, b=hh, prev via meta->prev_pool_idx.
                     z = sigmoid(ih_z + hh_z), r = sigmoid(ih_r + hh_r)
                     n = tanh(ih_n + r * hh_n)
                     h' = (1-z)*n + z*prev                                 */
                auto meta = (GruCellReplayMeta*)e.meta;
                int oo = meta->o;
                auto prev = pool[meta->prev_pool_idx];
                auto ih_z = mx::slice(a, {0}, {oo});
                auto ih_r = mx::slice(a, {oo}, {2*oo});
                auto ih_n = mx::slice(a, {2*oo}, {3*oo});
                auto hh_z = mx::slice(b, {0}, {oo});
                auto hh_r = mx::slice(b, {oo}, {2*oo});
                auto hh_n = mx::slice(b, {2*oo}, {3*oo});
                auto z = mx::sigmoid(mx::add(ih_z, hh_z));
                auto r_gate = mx::sigmoid(mx::add(ih_r, hh_r));
                auto n = mx::tanh(mx::add(ih_n, mx::multiply(r_gate, hh_n)));
                pool[out] = mx::add(mx::multiply(mx::subtract(one_like(z), z), n),
                                    mx::multiply(z, prev));
                break;
            }
            case OP_EMBEDDING: {
                // a = weight, b = indices (int32), scalar_arg = embedDim
                auto idx_int = mx::astype(b, mx::int32);
                auto rows = mx::take(a, idx_int, 0);
                pool[out] = mx::flatten(rows);
                break;
            }
            case OP_BATCH_NORM: {
                auto* bm = (BatchNormReplayMeta*)e.meta;
                auto x = mx::reshape(a, {bm->C, bm->spatial});
                auto mean = mx::mean(x, std::vector<int>{1}, true);
                auto var = mx::var(x, std::vector<int>{1}, true);
                auto rstd = mx::rsqrt(mx::add(var, scalar_like(bm->eps, var)));
                auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
                auto g = mx::reshape(pool[bm->gamma_pool_idx], {bm->C, 1});
                auto bt = mx::reshape(pool[bm->beta_pool_idx], {bm->C, 1});
                pool[out] = mx::flatten(mx::add(mx::multiply(g, x_hat), bt));
                break;
            }
            case OP_DROPOUT: {
                // b holds the stored mask tensor; just multiply
                pool[out] = mx::multiply(a, b);
                break;
            }
            case OP_AVG_POOL1D: {
                // scalar_arg encodes kL + stride*0.001
                int kL = (int)e.scalar_arg;
                int stride = (int)((e.scalar_arg - kL) * 1000 + 0.5);
                if (stride == 0) stride = kL;
                int oL = ((int)a.shape(1) - kL) / stride + 1;
                mx::array res = mx::zeros({(int)a.shape(0), oL}, a.dtype());
                for (int kl = 0; kl < kL; kl++) {
                    auto sliced = mx::slice(a, {0, kl}, {(int)a.shape(0), kl + oL*stride}, {1, stride});
                    res = mx::add(res, sliced);
                }
                pool[out] = mx::divide(res, scalar_like((double)kL, a));
                break;
            }
            case OP_AVG_POOL2D: {
                // For simplicity, re-derive dims from input shape. Only k=2 s=2 common case tested.
                int CC = (int)a.shape(0), HH = (int)a.shape(1), WW = (int)a.shape(2);
                // Default: k=2, stride=2 (most common usage)
                int kH = 2, kW = 2, sH = 2, sW = 2;
                int oH = (HH - kH)/sH + 1, oW = (WW - kW)/sW + 1;
                mx::array res = mx::zeros({CC, oH, oW}, a.dtype());
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        auto sl = mx::slice(a, {0,kh,kw}, {CC,kh+oH*sH,kw+oW*sW}, {1,sH,sW});
                        res = mx::add(res, sl);
                    }
                pool[out] = mx::divide(res, scalar_like((double)(kH*kW), a));
                break;
            }
            case OP_CONV1D: {
                auto* cm = (Conv1DReplayMeta*)e.meta;
                int inC = cm->inC, LL = cm->L;
                auto inp_lc = mx::transpose(a, {1, 0});
                auto inp_nlc = mx::reshape(inp_lc, {1, LL, inC});
                auto ker_mlx = mx::transpose(b, {0, 2, 1});
                auto cv = mx::conv1d(inp_nlc, ker_mlx, cm->stride, cm->pad);
                auto cv_sq = mx::squeeze(cv, 0);
                auto cv_out = mx::transpose(cv_sq, {1, 0});
                if (cm->bias_pool_idx >= 0)
                    cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1}));
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL1D: {
                auto* pm = (MaxPool1DReplayMeta*)e.meta;
                mx::array res = mx::full({pm->C, pm->oL}, -1e30, a.dtype());
                for (int kl = 0; kl < pm->kL; kl++) {
                    auto sliced = mx::slice(a, {0, kl}, {pm->C, kl + pm->oL * pm->stride}, {1, pm->stride});
                    res = mx::maximum(res, sliced);
                }
                pool[out] = res;
                break;
            }
            case OP_CONV2D: {
                auto* cm = (Conv2DReplayMeta*)e.meta;
                int inC = cm->inC, HH = cm->H, WW = cm->W;
                auto inp_hwc = mx::transpose(a, {1, 2, 0});
                auto inp_nhwc = mx::reshape(inp_hwc, {1, HH, WW, inC});
                auto ker_mlx = mx::transpose(b, {0, 2, 3, 1});
                auto cv = mx::conv2d(inp_nhwc, ker_mlx,
                                     {cm->strH, cm->strW}, {cm->padH, cm->padW});
                auto cv_sq = mx::squeeze(cv, 0);
                auto cv_out = mx::transpose(cv_sq, {2, 0, 1});
                if (cm->bias_pool_idx >= 0) {
                    cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1, 1}));
                }
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL2D: {
                auto* pm = (MaxPool2DReplayMeta*)e.meta;
                mx::array res = mx::full({pm->C, pm->oH, pm->oW}, -1e30, a.dtype());
                for (int kh = 0; kh < pm->kH; kh++) {
                    for (int kw = 0; kw < pm->kW; kw++) {
                        auto sliced = mx::slice(a,
                            {0, kh, kw},
                            {pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
                            {1, pm->strH, pm->strW});
                        res = mx::maximum(res, sliced);
                    }
                }
                pool[out] = res;
                break;
            }
            case OP_CONV2D_BATCHED: {
                auto* cm = (Conv2DBatchedReplayMeta*)e.meta;
                int B = cm->B, inC = cm->inC, HH = cm->H, WW = cm->W;
                (void)inC; (void)HH; (void)WW;  // dimensions inferred from shape
                auto inp_nhwc = mx::transpose(a, {0, 2, 3, 1});
                auto ker_mlx  = mx::transpose(b, {0, 2, 3, 1});
                auto cv = mx::conv2d(inp_nhwc, ker_mlx,
                                     {cm->strH, cm->strW}, {cm->padH, cm->padW});
                auto cv_out = mx::transpose(cv, {0, 3, 1, 2});
                if (cm->bias_pool_idx >= 0) {
                    cv_out = mx::add(cv_out,
                                     mx::reshape(pool[cm->bias_pool_idx], {1, -1, 1, 1}));
                }
                (void)B;
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL2D_BATCHED: {
                auto* pm = (MaxPool2DBatchedReplayMeta*)e.meta;
                mx::array res = mx::full({pm->B, pm->C, pm->oH, pm->oW}, -1e30, a.dtype());
                for (int kh = 0; kh < pm->kH; kh++) {
                    for (int kw = 0; kw < pm->kW; kw++) {
                        auto sliced = mx::slice(a,
                            {0, 0, kh, kw},
                            {pm->B, pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
                            {1, 1, pm->strH, pm->strW});
                        res = mx::maximum(res, sliced);
                    }
                }
                pool[out] = res;
                break;
            }
            case OP_CUMPROD: {
                pool[out] = mx::cumprod(a, 0);
                break;
            }
            case OP_GATHER: {
                // Indices are discrete and non-differentiable — read directly
                // from the tape entry's tensor (closure-captured, not via
                // pool). The constants-collection above intentionally
                // excludes arg2 for this op so mlx::vjp never sees it as a
                // differentiable input. See `arg2_is_index` above.
                auto idx_int = mx::astype(e.arg2->data, mx::int32);
                pool[out] = mx::take(a, idx_int, 0);
                break;
            }
            case OP_SCATTER_ADD: {
                int out_size = (int)e.scalar_arg;
                auto idx_int = mx::astype(e.arg2->data, mx::int32);
                auto base = mx::zeros({out_size}, a.dtype());
                auto updates_2d = mx::reshape(a, {(int)a.size(), 1});
                pool[out] = mx::scatter_add(base, {idx_int}, updates_2d, std::vector<int>{0});
                break;
            }
            default: break;
            }
        }
        return pool[loss_pool_idx];
    };

    // Compute gradients via MLX native autograd (vjp with unit cotangent)
    auto forward_vec = [&](const std::vector<mx::array>& xs) -> std::vector<mx::array> {
        return {forward_fn(xs)};
    };

    // Build the [params..., constants...] inputs vector
    std::vector<mx::array> all_inputs;
    all_inputs.reserve(n_params + n_consts);
    for (auto& p : param_arrays) all_inputs.push_back(p);
    for (auto& [idx, arr] : constants) all_inputs.push_back(arr);

    // Job 3 Phase B — compile-enabled path. Stage 4 wires mx::compile
    // for real. The compile call is the public C++ overload; until we
    // add caching (Stage 5+), it recompiles every backward.
    std::pair<std::vector<mx::array>, std::vector<mx::array>> vjp_result;
    if (tensor_mlx_compile_enabled()) {
        g_compile_invocations++;
        auto compiled = mx::compile(forward_vec);
        vjp_result = mx::vjp(compiled, all_inputs, {mx::array(1.0f)});
    } else {
        vjp_result = mx::vjp(forward_vec, all_inputs, {mx::array(1.0f)});
    }
    // vjp returned grads for [params..., constants...]; truncate to params.
    // mx::array has no default ctor, so erase the tail rather than resize.
    auto& grads = vjp_result.second;
    if ((int)grads.size() > n_params)
        grads.erase(grads.begin() + n_params, grads.end());

    // Distribute gradients to parameter tensors
    for (int i = 0; i < param_count(); i++) {
        auto* tensor = (Tensor*)param_tensor(i);
        tensor->grad = grads[i];
        tensor->has_grad = true;
    }

    // Optional NaN trap — fires only when DEBUG_NAN_TRAP=1 in the env.
    // Walks every param grad on first appearance of NaN/Inf and logs the
    // offending param name. Useful to localise gradient blow-up at the
    // peaked-attention working point in NTM/DNC training.
    {
        static int reported = 0;
        const char* env = getenv("DEBUG_NAN_TRAP");
        if (env && env[0] == '1' && !reported) {
            int any_nan = 0;
            for (int i = 0; i < param_count(); i++) {
                const char* p_name = param_name(i);
                auto* p_tensor = (Tensor*)param_tensor(i);
                auto contig = mx::contiguous(p_tensor->grad);
                mx::eval(contig);
                long n = (long)contig.size();
                std::vector<double> buf((size_t)n);
                mx_to_doubles(contig, buf.data());
                const double* gp = buf.data();
                int nan_count = 0, inf_count = 0;
                double maxabs = 0.0;
                for (long j = 0; j < n; j++) {
                    double v = gp[j];
                    if (v != v) nan_count++;
                    else if (v > 1e30 || v < -1e30) inf_count++;
                    else { double a = v < 0 ? -v : v; if (a > maxabs) maxabs = a; }
                }
                if (nan_count || inf_count) {
                    fprintf(stderr, "[NAN_TRAP] param[%d]=%s NaN=%d Inf=%d maxabs=%.3e (n=%ld)\n",
                            i, p_name, nan_count, inf_count, maxabs, n);
                    any_nan = 1;
                }
            }
            // If any param grad is bad, walk the forward tape and find the
            // first NaN-producing op. result->data already holds the actual
            // forward value, so we just check those in tape order.
            if (any_nan) {
                static const char* OP_NAMES[] = {
                    "CONST", "ADD", "SUB", "MUL", "DIV", "NEG", "EXP", "LOG", "SQRT",
                    "SIGMOID", "TANH", "ADD_SCALAR", "MUL_SCALAR", "CLAMP_MIN",
                    "SUM", "MEAN", "MM", "BMM", "TRANSPOSE_2D", "SOFTMAX_2D",
                    "LOG_SOFTMAX_2D", "MASKED_FILL", "LAYER_NORM_2D", "RESHAPE",
                    "NARROW", "CAT", "POW", "ABS", "STACK", "OUTER", "COSINE_SIM",
                    "CONV1D_CIRC", "MV", "SELECT", "BMM_3X3", "SOFTMAX_3D",
                    "TRANSPOSE_LAST2", "GELU", "GRU_CELL", "EMBEDDING", "BATCH_NORM",
                    "DROPOUT", "AVG_POOL1D", "AVG_POOL2D", "CONV1D", "MAX_POOL1D",
                    "CONV2D", "MAX_POOL2D", "CUMPROD", "GATHER", "SCATTER_ADD",
                    "LEAKY_RELU", "SILU", "SUM_DIM", "CAT_MULTI", "LINEAR_2D",
                    "CONCAT_2D_AXIS1", "SOFTPLUS",
                };
                int n_names = sizeof(OP_NAMES) / sizeof(OP_NAMES[0]);
                fprintf(stderr, "[NAN_TRAP] scanning forward tape (size=%d) for first NaN op...\n",
                        (int)tape.size());
                for (int i = 0; i < (int)tape.size(); i++) {
                    auto& e = tape[i];
                    if (!e.result) continue;
                    auto contig = mx::contiguous(e.result->data);
                    mx::eval(contig);
                    long n = (long)contig.size();
                    if (n == 0) continue;
                    std::vector<double> r_buf((size_t)n);
                    mx_to_doubles(contig, r_buf.data());
                    const double* dp = r_buf.data();
                    int nan_count = 0;
                    for (long j = 0; j < n; j++) {
                        double v = dp[j];
                        if (v != v) { nan_count++; }
                    }
                    if (nan_count) {
                        const char* opn = (e.op >= 0 && e.op < n_names)
                            ? OP_NAMES[e.op] : "UNKNOWN";
                        fprintf(stderr, "[NAN_TRAP] first NaN at tape[%d] op=%s (id=%d) result.size=%ld nan_count=%d arg1.op=%d arg2.op=%d\n",
                                i, opn, e.op, n, nan_count,
                                e.arg1 ? (int)tape[e.arg1->tape_idx].op : -1,
                                e.arg2 ? (int)tape[e.arg2->tape_idx].op : -1);
                        // Sample arg1/arg2 values to spot inputs that are
                        // already large/small.
                        if (e.arg1) {
                            auto a = mx::contiguous(e.arg1->data);
                            mx::eval(a);
                            std::vector<double> a_buf((size_t)a.size());
                            mx_to_doubles(a, a_buf.data());
                            const double* ap = a_buf.data();
                            double amin = ap[0], amax = ap[0];
                            int anan = 0;
                            for (long j = 0; j < (long)a.size(); j++) {
                                double v = ap[j];
                                if (v != v) anan++;
                                else { if (v < amin) amin = v; if (v > amax) amax = v; }
                            }
                            fprintf(stderr, "[NAN_TRAP]   arg1 size=%ld nan=%d range=[%.3e, %.3e]\n",
                                    (long)a.size(), anan, amin, amax);
                        }
                        if (e.arg2) {
                            auto b = mx::contiguous(e.arg2->data);
                            mx::eval(b);
                            std::vector<double> b_buf((size_t)b.size());
                            mx_to_doubles(b, b_buf.data());
                            const double* bp = b_buf.data();
                            double bmin = bp[0], bmax = bp[0];
                            int bnan = 0;
                            for (long j = 0; j < (long)b.size(); j++) {
                                double v = bp[j];
                                if (v != v) bnan++;
                                else { if (v < bmin) bmin = v; if (v > bmax) bmax = v; }
                            }
                            fprintf(stderr, "[NAN_TRAP]   arg2 size=%ld nan=%d range=[%.3e, %.3e]\n",
                                    (long)b.size(), bnan, bmin, bmax);
                        }
                        reported = 1;
                        break;
                    }
                }
            }
            if (reported) fflush(stderr);
        }
    }

    prof_backward_ms_mlx += _wall_ms_mlx() - t0_bwd;
}

TensorHandle tensor_grad(TensorHandle h) {
    auto t = (Tensor*)h;
    if (!t->has_grad) return nullptr;
    /* mx::vjp may return non-contiguous grads (broadcast strides). Force
       a contiguous copy so the returned tensor has the expected layout. */
    auto contig = mx::contiguous(t->grad);
    mx::eval(contig);
    return (TensorHandle)new Tensor(contig, false);
}

void tensor_zero_grad(TensorHandle h) {
    auto t = (Tensor*)h;
    if (t->has_grad) {
        t->grad = mx::zeros(t->data.shape(), t->data.dtype());
    }
}

int tensor_requires_grad(TensorHandle h) { return ((Tensor*)h)->requires_grad ? 1 : 0; }
extern "C" TensorHandle tensor_detach_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* Detach: clone data, requires_grad=false, no tape entry. The result is
       a leaf with no autograd linkage to the source tensor. */
    auto t = (Tensor*)h;
    return (TensorHandle)new Tensor(mx::array(t->data), false);

}
TensorHandle tensor_detach(TensorHandle h) {
    return tensor_detach_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_with_grad_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);

    /* Promote a tensor into the autograd graph: clone with requires_grad=true,
       record an OP_CONST tape entry so the constant pool picks up its data
       during backward replay. Note: for the result's gradient to actually be
       computed, the caller still needs to register it via param_register. */
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::array(t->data), true);
    tape_append(OP_CONST, r, nullptr, nullptr, 0);
    return (TensorHandle)r;

}
TensorHandle tensor_with_grad(TensorHandle h) {
    return tensor_with_grad_mlx_streamed(h, default_stream_tag());
}

void tensor_set_requires_grad(TensorHandle h, int rg) {
    ((Tensor*)h)->requires_grad = (rg != 0);
}

// Generation-scoped sweep: eval pending lazy graphs, then delete every
// wrap-only (rc==1) Tensor created at or after `block_start`. Those are
// block/epoch-local intermediates whose results have been extracted to
// scalars or retained (rc>=2) by KeepAlive. Registry params (rc>1) and
// pre-generation state (lower create_id) are spared. Bounds the live
// handle / Metal-buffer count instead of letting it accumulate past the
// paravirt-Metal ceiling. Shared by no_grad_end and epoch_end.
static void mlx_sweep_generation(long block_start) {
    std::vector<mx::array> to_eval;
    for (auto* t : all_tensors) to_eval.push_back(t->data);
    if (!to_eval.empty()) {
        try { mx::eval(to_eval); } catch (...) { /* best-effort */ }
    }
    std::vector<Tensor*> survivors;
    survivors.reserve(all_tensors.size());
    for (auto* t : all_tensors) {
        if (t->refcount == 1 && t->create_id >= block_start) {
            // Wrap-only block-local intermediate. We must NOT `delete` it:
            // rc==1 here is the Idris guardian wrap's own retain, and that
            // wrap is still registered (un-drained — a drained wrap would
            // have dropped this to rc==0). Its eventual drain calls
            // tensor_release_handle on this exact pointer; freeing the
            // object now makes that an unguarded `refcount--` on freed (and,
            // under allocation churn, recycled) memory → malloc-freelist
            // corruption (the intermittent F32 mlx-gpu SIGTRAP). Instead
            // release the heavy mx::array buffers now — that reclaims the
            // Metal MTLBuffer, which is the only thing the live-handle
            // ceiling actually cares about — and keep the lightweight husk
            // alive (its address pinned) until the wrap drains it to rc==0,
            // when the branch below frees it safely.
            //
            // Assign a single process-wide empty scalar rather than a fresh
            // `mx::array(0.0f)` per husk: mx::array is copy-on-write (a
            // shared_ptr to its buffer), so this is a refcount bump, not an
            // allocation, yet it still drops the husk's heavy buffer. A fresh
            // per-husk scalar *does* allocate, and on Apple Silicon every
            // buffer — even 4–8 bytes — routes through MetalAllocator; under
            // the paravirt-Metal MTLBuffer ceiling (Tart/GHA VMs) those
            // per-sweep allocations throw `[malloc] Unable to allocate N
            // bytes` mid-training (regression from 8482788, NTM/DNC/mnist/RL).
            static const mx::array g_husk_empty = mx::array(0.0f);
            t->data = g_husk_empty;
            t->grad = g_husk_empty;
            t->has_grad = false;
            survivors.push_back(t);
            continue;
        }
        if (t->refcount > 0) survivors.push_back(t);
        else delete t;
    }
    all_tensors = std::move(survivors);
    try { mx::clear_cache(); } catch (...) { /* best-effort */ }
}

static long g_nograd_block_start = 0;  // create_id at outermost no_grad_begin
void tensor_no_grad_begin(void) {
    if (no_grad_depth == 0) g_nograd_block_start = g_mlx_create_calls_global;
    no_grad_depth++;
}
void tensor_no_grad_end(void) {
    if (no_grad_depth > 0) no_grad_depth--;
    if (no_grad_depth > 0) return;  // only sweep on outermost end
    mlx_sweep_generation(g_nograd_block_start);
}

// Generation-scoped free for grad-mode training, nestable via a marker
// stack: the per-epoch bracket (runTrainingIO) is the outer frame and a
// per-step `withGenFree` bracket is an inner frame. begin pushes the
// current create_id; end pops it and frees wrap-only handles created since.
static std::vector<long> g_gen_stack;
void tensor_epoch_begin(void) { g_gen_stack.push_back(g_mlx_create_calls_global); }
void tensor_epoch_end(void) {
    if (g_gen_stack.empty()) return;
    long start = g_gen_stack.back();
    g_gen_stack.pop_back();
    mlx_sweep_generation(start);
}

/* ================================================================
   Device
   ================================================================ */

TensorHandle tensor_to_device(TensorHandle t, const char* device) { return t; }
const char* tensor_device(TensorHandle t) { return "gpu"; }

/* ================================================================
   LSTM (stubs)
   ================================================================ */

void tensor_lstm_cell(TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh, TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c) {
    /* 1D variant: combined = w_ih @ input + b_ih + w_hh @ hx + b_hh
       Then dispatches to lstm_gates for the gate split + cell update.
       Each sub-op records its own tape entry; backward flows automatically. */
    int hidden = (int)((Tensor*)cx)->data.size();
    TensorHandle gi = tensor_mv(w_ih, input);
    TensorHandle gi_b = tensor_add(gi, b_ih);
    TensorHandle gh = tensor_mv(w_hh, hx);
    TensorHandle gh_b = tensor_add(gh, b_hh);
    TensorHandle combined = tensor_add(gi_b, gh_b);
    tensor_lstm_gates(combined, cx, hidden, out_h, out_c);
}

void tensor_lstm_gates(TensorHandle combined, TensorHandle prev_cell, int o,
    TensorHandle* out_h, TensorHandle* out_c) {
    /* Void-output variant: same decomposition as tensor_lstm_gates_pair, but
       returns through out_h/out_c pointers instead of a TensorPair.
       Delegate to _pair to share the implementation. */
    TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
    *out_h = p->first;
    *out_c = p->second;
    /* The pair struct itself is tracked in all_pairs and cleaned up at
       tape_reset. The caller doesn't own it; the outputs are the standalone
       Tensor handles inside. */
}
extern "C" TensorPair* tensor_lstm_gates_pair_mlx_streamed(TensorHandle hcombined, TensorHandle hprev_cell, int o, int stream_tag) {
    WITH_STREAM(stream_tag);

    // Decompose into primitives — each records its own tape entry.
    // Thread stream_tag through every inner call so the type-level device
    // stays in effect (the unsuffixed sub-op trampolines would each open
    // their own WITH_STREAM(default_stream_tag()) and clobber our scope).
    // Split combined [4*o] into 4 gates
    TensorHandle ig_raw = tensor_narrow_mlx_streamed(hcombined, 0, 0, o, stream_tag);
    TensorHandle fg_raw = tensor_narrow_mlx_streamed(hcombined, 0, o, o, stream_tag);
    TensorHandle gg_raw = tensor_narrow_mlx_streamed(hcombined, 0, 2*o, o, stream_tag);
    TensorHandle og_raw = tensor_narrow_mlx_streamed(hcombined, 0, 3*o, o, stream_tag);
    // Apply activations
    TensorHandle ig = tensor_sigmoid_mlx_streamed(ig_raw, stream_tag);
    TensorHandle fg = tensor_sigmoid_mlx_streamed(fg_raw, stream_tag);
    TensorHandle gg = tensor_tanh_mlx_streamed(gg_raw, stream_tag);
    TensorHandle og = tensor_sigmoid_mlx_streamed(og_raw, stream_tag);
    // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    TensorHandle fc = tensor_mul_mlx_streamed(fg, hprev_cell, stream_tag);
    TensorHandle ig_gg = tensor_mul_mlx_streamed(ig, gg, stream_tag);
    TensorHandle new_cell = tensor_add_mlx_streamed(fc, ig_gg, stream_tag);
    // h_t = o_t ⊙ tanh(c_t)
    TensorHandle tanh_cell = tensor_tanh_mlx_streamed(new_cell, stream_tag);
    TensorHandle new_hidden = tensor_mul_mlx_streamed(og, tanh_cell, stream_tag);
    // Return pair
    auto pair = (TensorPair*)malloc(sizeof(TensorPair));
    pair->first = new_hidden;
    pair->second = new_cell;
    all_pairs.push_back(pair);
    return pair;

}
TensorPair* tensor_lstm_gates_pair(TensorHandle hcombined, TensorHandle hprev_cell, int o) {
    return tensor_lstm_gates_pair_mlx_streamed(hcombined, hprev_cell, o, default_stream_tag());
}

extern "C" TensorHandle tensor_pair_first_mlx_streamed(TensorPair* p, int stream_tag) {
    WITH_STREAM(stream_tag);
 return p->first; 
}
TensorHandle tensor_pair_first(TensorPair* p) {
    return tensor_pair_first_mlx_streamed(p, default_stream_tag());
}
extern "C" TensorHandle tensor_pair_second_mlx_streamed(TensorPair* p, int stream_tag) {
    WITH_STREAM(stream_tag);
 return p->second; 
}
TensorHandle tensor_pair_second(TensorPair* p) {
    return tensor_pair_second_mlx_streamed(p, default_stream_tag());
}
void tensor_pair_free(TensorPair* p) { if (p) free(p); }

/* ================================================================
   Parameter registry — surface lifted into shared/training/param_registry.c.

   Routes through `g_active_port_mlx` for per-tensor accesses (numel,
   grad-read/write, zero, bulk load). The shared registry's
   tensor_retain/release wrap each register/clear so mlx's refcount
   lifecycle (where the registry contributes +1 to keep params alive
   against the all_tensors sweep) is preserved. The shared param_clear
   no longer triggers tape_reset — that's covered by optimizer_step
   and backend_reset_for_eval. */

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    auto t = (Tensor*)h;
    t->data = mx::subtract(t->data, scalar_like(val, t->data));
    return h;
}

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

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    std::vector<mx::array> arrs;
    bool rg = false;
    for (int i = 0; i < count; i++) {
        auto t = (Tensor*)arr[i];
        arrs.push_back(t->data);
        if (t->requires_grad) rg = true;
    }
    auto r = new Tensor(mx::stack(arrs, dim), rg);
    /* Record OP_STACK with scalar_arg=dim and meta=input pool indices.
       Replay reads dim from scalar_arg so non-zero stack dims backprop correctly. */
    if (rg) {
        int idx = tape_append(OP_STACK, r, nullptr, nullptr, (double)dim);
        auto* indices = new std::vector<int>();
        for (int i = 0; i < count; i++)
            indices->push_back(((Tensor*)arr[i])->pool_idx);
        tape[idx].meta = (void*)indices;
    }
    /* Caller (Idris) allocates arr via prim__ptrArrayAlloc; tape and torch
       both free it after consuming. MLX matches that convention. */
    free(arr);
    return (TensorHandle)r;
}

TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    std::vector<mx::array> arrs;
    bool rg = false;
    for (int i = 0; i < count; i++) {
        auto t = (Tensor*)arr[i];
        arrs.push_back(t->data);
        if (t->requires_grad) rg = true;
    }
    auto r = new Tensor(mx::concatenate(arrs, dim), rg);
    if (rg) {
        int idx = tape_append(OP_CAT_MULTI, r, nullptr, nullptr, (double)dim);
        auto* indices = new std::vector<int>();
        for (int i = 0; i < count; i++)
            indices->push_back(((Tensor*)arr[i])->pool_idx);
        tape[idx].meta = (void*)indices;
    }
    /* Match torch convention: caller passes ownership of arr (allocated via
       tensor_ptr_array_alloc), we free it after consuming. */
    free(arr);
    return (TensorHandle)r;
}

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

double tensor_item_2d(TensorHandle mat, int row, int col) {
    auto t = (Tensor*)mat;
    // Flatten to contiguous for correct indexing on non-contiguous views (e.g. transpose)
    auto flat = mx::flatten(t->data, mx::StreamOrDevice{});
    mx::eval(flat);
    int cols = t->data.shape(1);
    return mx_read_double(flat, (long)row * cols + col);
}

extern "C" double tensor_item_1d_mlx_streamed(TensorHandle vec, int idx, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)vec;
    mx::eval(t->data);
    return mx_read_double(t->data, idx);
}
double tensor_item_1d(TensorHandle vec, int idx) {
    return tensor_item_1d_mlx_streamed(vec, idx, default_stream_tag());
}

/* ================================================================
   Optimizer
   ================================================================ */

struct Optimizer {
    int type; // 0=sgd, 1=rmsprop, 2=adam
    double lr, beta1, beta2, eps;
    double alpha, weight_decay, momentum;
    int t;
    // Per-parameter buffers
    std::vector<mx::array> m_bufs, v_bufs;
    // Per-param LR overrides (indexed by param registry position, -1 = use base)
    std::vector<double> param_lr;
    std::string prefix;  // empty = manages all params; else prefix filter
};

/* Returns true if param[i]'s name starts with opt->prefix (or prefix is empty). */
static bool opt_owns_param_mlx(Optimizer* opt, int i) {
    if (opt->prefix.empty()) return true;
    return std::string(param_name(i)).rfind(opt->prefix, 0) == 0;
}

OptimizerHandle optimizer_create_sgd(double lr) {
    auto opt = new Optimizer();
    opt->type = 0; opt->lr = lr; opt->t = 0;
    return (OptimizerHandle)opt;
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
    double weight_decay, double momentum) {
    auto opt = new Optimizer();
    opt->type = 1; opt->lr = lr; opt->alpha = alpha; opt->eps = eps;
    opt->weight_decay = weight_decay; opt->momentum = momentum; opt->t = 0;
    return (OptimizerHandle)opt;
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    auto opt = new Optimizer();
    opt->type = 2; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps; opt->t = 0;
    return (OptimizerHandle)opt;
}

OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                            double eps, const char* prefix) {
    auto opt = new Optimizer();
    opt->type = 2; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps; opt->t = 0;
    opt->prefix = prefix ? std::string(prefix) : std::string();
    return (OptimizerHandle)opt;
}

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
    auto opt = new Optimizer();
    opt->type = 3; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2;
    opt->eps = eps; opt->weight_decay = weight_decay; opt->t = 0;
    return (OptimizerHandle)opt;
}

void optimizer_free(OptimizerHandle h) { delete (Optimizer*)h; }
void optimizer_zero_grad(OptimizerHandle h) { param_zero_all_grads(); }

void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
    auto opt = (Optimizer*)h;
    int np = param_count();
    if ((int)opt->param_lr.size() < np)
        opt->param_lr.resize(np, -1.0);
    for (int i = 0; i < np; i++) {
        if (strcmp(param_name(i), name) == 0) {
            opt->param_lr[i] = lr;
            return;
        }
    }
}

void optimizer_set_lr(OptimizerHandle h, double lr) {
    auto opt = (Optimizer*)h;
    opt->lr = lr;
}

static void _dbg_dump_param_grads_if_enabled_mlx(void) {
    static int dumped = 0;
    static int max_dumps = -1;
    if (max_dumps < 0) {
        const char* mx_env = getenv("DEBUG_PARAM_GRADS_MAX");
        max_dumps = mx_env ? atoi(mx_env) : 1;
    }
    const char* env = getenv("DEBUG_PARAM_GRADS");
    if (!env || env[0] != '1') return;
    if (dumped >= max_dumps) return;
    dumped++;
    fprintf(stderr, "[DEBUG_PARAM_GRADS_MLX] dump #%d (np=%d):\n",
            dumped, param_count());
    for (int i = 0; i < param_count(); i++) {
        const char* p_name = param_name(i);
        auto t = (Tensor*)param_tensor(i);
        long n = (long)t->data.size();
        double l2 = 0.0;
        int has_grad = t->has_grad ? 1 : 0;
        int rg = t->requires_grad ? 1 : 0;
        if (t->has_grad) {
            mx::eval(t->grad);
            auto contig = mx::contiguous(t->grad);
            mx::eval(contig);
            std::vector<double> gbuf((size_t)n);
            mx_to_doubles(contig, gbuf.data());
            const double* gp = gbuf.data();
            for (long j = 0; j < n; j++) l2 += gp[j] * gp[j];
        }
        l2 = sqrt(l2);
        fprintf(stderr, "  [%d] %s (n=%ld rg=%d hg=%d) grad_l2=%.6e\n",
                i, p_name, n, rg, has_grad, l2);
    }
    fflush(stderr);
}

static bool mlx_opt_compile_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("MLX_OPT_COMPILE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached == 1;
}

/* Adam step compiled via mx::compile.
   Layout of inputs vector for the compiled function:
     [0 .. N-1]            params (current values)
     [N .. 2N-1]            grads
     [2N .. 3N-1]           m buffers (exp_avg)
     [3N .. 4N-1]           v buffers (exp_avg_sq)
     [4N .. 5N-1]           per-param learning rates (scalar arrays)
     [5N + 0 .. 5N + 6]     beta1, 1-beta1, beta2, 1-beta2, bc1, bc2, eps
   Outputs:
     [0 .. N-1]             new params
     [N .. 2N-1]            new m
     [2N .. 3N-1]           new v

   We keep the compiled function in a static map keyed on N (the active param
   count). mlx caches by input-shape signature internally, so repeated calls
   with the same shape tuple hit the trace cache after first invocation. The
   function lambda must be defined once per N — recreating it per call would
   miss mlx's identity-based cache. */
static std::unordered_map<int, std::function<std::vector<mx::array>(
    const std::vector<mx::array>&)>> adam_compiled_by_n;

static std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
get_adam_compiled(int n) {
    auto it = adam_compiled_by_n.find(n);
    if (it != adam_compiled_by_n.end()) return it->second;
    auto raw = [n](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
        const mx::array& beta1  = ins[5*n + 0];
        const mx::array& one_b1 = ins[5*n + 1];
        const mx::array& beta2  = ins[5*n + 2];
        const mx::array& one_b2 = ins[5*n + 3];
        const mx::array& bc1    = ins[5*n + 4];
        const mx::array& bc2    = ins[5*n + 5];
        const mx::array& eps    = ins[5*n + 6];
        std::vector<mx::array> new_p, new_m, new_v;
        new_p.reserve(n); new_m.reserve(n); new_v.reserve(n);
        for (int i = 0; i < n; i++) {
            const mx::array& p  = ins[i];
            const mx::array& g  = ins[n + i];
            const mx::array& m  = ins[2*n + i];
            const mx::array& v  = ins[3*n + i];
            const mx::array& lr = ins[4*n + i];
            auto m_n = mx::add(mx::multiply(beta1, m), mx::multiply(one_b1, g));
            auto v_n = mx::add(mx::multiply(beta2, v),
                               mx::multiply(one_b2, mx::square(g)));
            auto mhat = mx::divide(m_n, bc1);
            auto vhat = mx::divide(v_n, bc2);
            auto p_n = mx::subtract(p,
                mx::divide(mx::multiply(lr, mhat),
                           mx::add(mx::sqrt(vhat), eps)));
            new_p.push_back(p_n);
            new_m.push_back(m_n);
            new_v.push_back(v_n);
        }
        std::vector<mx::array> outs;
        outs.reserve(3*n);
        for (auto& a : new_p) outs.push_back(a);
        for (auto& a : new_m) outs.push_back(a);
        for (auto& a : new_v) outs.push_back(a);
        return outs;
    };
    adam_compiled_by_n[n] = mx::compile(raw);
    return adam_compiled_by_n[n];
}

static void adam_step_compile(Optimizer* opt, int np) {
    /* Gather active params (must have grads) and corresponding state. */
    std::vector<int> active_idx;
    active_idx.reserve(np);
    std::vector<mx::array> ins;
    /* Will fill in order [p..., g..., m..., v..., lr..., scalars(7)]. */
    std::vector<mx::array> params_in, grads_in, ms_in, vs_in, lrs_in;
    params_in.reserve(np); grads_in.reserve(np);
    ms_in.reserve(np); vs_in.reserve(np); lrs_in.reserve(np);
    /* Pin the compiled-step dtype to the first eligible param's dtype. The
       traced graph mixes scalars + arrays, so all scalar inputs need to
       match the param dtype to avoid implicit promotion baking the wrong
       output type into the cached compile. */
    mx::Dtype compile_dtype = mx::float32;
    bool found_dtype = false;
    for (int i = 0; i < np; i++) {
        if (!opt_owns_param_mlx(opt, i)) continue;
        auto t = (Tensor*)param_tensor(i);
        if (!t->has_grad) continue;
        if (!found_dtype) { compile_dtype = t->data.dtype(); found_dtype = true; }
        double lr = opt->lr;
        if (i < (int)opt->param_lr.size() && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];
        active_idx.push_back(i);
        params_in.push_back(t->data);
        grads_in.push_back(t->grad);
        ms_in.push_back(opt->m_bufs[i]);
        vs_in.push_back(opt->v_bufs[i]);
        lrs_in.push_back(mx::array(lr, compile_dtype));
    }
    int n = (int)active_idx.size();
    if (n == 0) return;
    ins.reserve(5*n + 7);
    for (auto& a : params_in) ins.push_back(a);
    for (auto& a : grads_in)  ins.push_back(a);
    for (auto& a : ms_in)     ins.push_back(a);
    for (auto& a : vs_in)     ins.push_back(a);
    for (auto& a : lrs_in)    ins.push_back(a);
    double bc1 = 1.0 - std::pow(opt->beta1, (double)opt->t);
    double bc2 = 1.0 - std::pow(opt->beta2, (double)opt->t);
    ins.push_back(mx::array(opt->beta1, compile_dtype));
    ins.push_back(mx::array(1.0 - opt->beta1, compile_dtype));
    ins.push_back(mx::array(opt->beta2, compile_dtype));
    ins.push_back(mx::array(1.0 - opt->beta2, compile_dtype));
    ins.push_back(mx::array(bc1, compile_dtype));
    ins.push_back(mx::array(bc2, compile_dtype));
    ins.push_back(mx::array(opt->eps, compile_dtype));

    auto& compiled = get_adam_compiled(n);
    auto outs = compiled(ins);

    /* Distribute results back to param_registry / optimizer state. */
    for (int i = 0; i < n; i++) {
        int idx = active_idx[i];
        ((Tensor*)param_tensor(idx))->data = outs[i];
        opt->m_bufs[idx] = outs[n + i];
        opt->v_bufs[idx] = outs[2*n + i];
    }
}

void optimizer_step(OptimizerHandle h) {
    double t0_opt = _wall_ms_mlx();
    auto opt = (Optimizer*)h;
    opt->t++;
    int np = param_count();
    _dbg_dump_param_grads_if_enabled_mlx();

    // Ensure optimizer buffers
    if ((int)opt->m_bufs.size() != np) {
        opt->m_bufs.clear();
        opt->v_bufs.clear();
        for (int i_ = 0; i_ < param_count(); i_++) {
            auto* p_tensor = (Tensor*)param_tensor(i_);
            opt->m_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
            opt->v_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
        }
    }

    /* Adam-only compile path: gate via MLX_OPT_COMPILE=1.
       Reuses the cached compiled function (one per param-count signature).
       Other optimizer types fall through to the per-op loop below. */
    if (opt->type == 2 && mlx_opt_compile_enabled()) {
        double tm0 = _wall_ms_mlx();
        adam_step_compile(opt, np);
        prof_optimizer_math_ms_mlx += _wall_ms_mlx() - tm0;
        std::vector<mx::array> to_eval;
        for (int i_ = 0; i_ < param_count(); i_++) {
            to_eval.push_back(((Tensor*)param_tensor(i_))->data);
        }
        mx::eval(to_eval);
        tape_reset();
        for (int i_ = 0; i_ < param_count(); i_++) {
            auto* p_tensor = (Tensor*)param_tensor(i_);
            p_tensor->tape_idx = -1;
            p_tensor->has_grad = false;
            tape_append(OP_CONST, p_tensor, nullptr, nullptr, 0);
        }
        prof_optimizer_ms_mlx += _wall_ms_mlx() - t0_opt;
        prof_epochs_mlx++;
        return;
    }

    double tm0 = _wall_ms_mlx();

    // Hoist optimizer-state scalars out of the per-param loop. These depend on
    // opt and the current step, not on which param — re-allocating them per
    // param added one graph node per param per step for nothing.
    //
    // Build the scalars in the dtype of the first eligible param so they don't
    // force a runtime promotion at every multiply. The common case is uniform
    // param dtype across the registry; mixed-dtype models would rely on mlx's
    // promotion rules at the multiply boundary (correct but slower). If you
    // need mixed-dtype support, split the optimizer by dtype.
    mx::Dtype opt_dtype = mx::float32;
    for (int i = 0; i < np; i++) {
        auto* p_tensor = (Tensor*)param_tensor(i);
        if (opt_owns_param_mlx(opt, i) && p_tensor->has_grad) {
            opt_dtype = p_tensor->data.dtype();
            break;
        }
    }
    auto alpha_arr   = mx::array(opt->alpha, opt_dtype);
    auto one_m_alpha = mx::array(1.0 - opt->alpha, opt_dtype);
    auto beta1_arr   = mx::array(opt->beta1, opt_dtype);
    auto one_m_beta1 = mx::array(1.0 - opt->beta1, opt_dtype);
    auto beta2_arr   = mx::array(opt->beta2, opt_dtype);
    auto one_m_beta2 = mx::array(1.0 - opt->beta2, opt_dtype);
    auto eps_arr     = mx::array(opt->eps, opt_dtype);
    auto momentum_a  = mx::array(opt->momentum, opt_dtype);
    auto bc1_arr     = mx::array(1.0 - std::pow(opt->beta1, opt->t), opt_dtype);
    auto bc2_arr     = mx::array(1.0 - std::pow(opt->beta2, opt->t), opt_dtype);

    for (int i = 0; i < np; i++) {
        if (!opt_owns_param_mlx(opt, i)) continue;
        auto t = (Tensor*)param_tensor(i);
        if (!t->has_grad) continue;

        /* Don't eval(t->grad) here — that's a per-param sync (293 params
           x ~1 ms kernel-launch wall = ~250 ms/ep on GPU; see the
           2026-05-14 GptLarge Phase 3 entry in perf-changes.md). The
           ops below take lazy mx::array inputs happily; the trailing
           mx::eval(to_eval) past the loop walks the dependency graph
           and pulls grads into the same batch as the param updates. */
        auto g = t->grad;

        /* Per-param LR: use override if set, otherwise base LR */
        double lr = opt->lr;
        if (i < (int)opt->param_lr.size() && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];
        auto lr_arr = scalar_like(lr, t->data);

        switch (opt->type) {
        case 0: // SGD
            t->data = mx::subtract(t->data, mx::multiply(lr_arr, g));
            break;
        case 1: { // RMSprop — keep lr OUTSIDE the momentum buffer to match
                  // torch.optim.RMSprop. Folding lr into the buffer coincides
                  // with PyTorch only at constant lr; under an LR schedule the
                  // buffer carries stale rates and diverges.
            opt->v_bufs[i] = mx::add(mx::multiply(alpha_arr, opt->v_bufs[i]),
                                      mx::multiply(one_m_alpha, mx::square(g)));
            auto avg = mx::add(mx::sqrt(opt->v_bufs[i]), eps_arr);
            if (opt->momentum > 0) {
                opt->m_bufs[i] = mx::add(mx::multiply(momentum_a, opt->m_bufs[i]),
                                          mx::divide(g, avg));
                t->data = mx::subtract(t->data, mx::multiply(lr_arr, opt->m_bufs[i]));
            } else {
                t->data = mx::subtract(t->data, mx::divide(mx::multiply(lr_arr, g), avg));
            }
            break;
        }
        case 2: { // Adam
            opt->m_bufs[i] = mx::add(mx::multiply(beta1_arr, opt->m_bufs[i]),
                                      mx::multiply(one_m_beta1, g));
            opt->v_bufs[i] = mx::add(mx::multiply(beta2_arr, opt->v_bufs[i]),
                                      mx::multiply(one_m_beta2, mx::square(g)));
            auto mhat = mx::divide(opt->m_bufs[i], bc1_arr);
            auto vhat = mx::divide(opt->v_bufs[i], bc2_arr);
            t->data = mx::subtract(t->data,
                mx::divide(mx::multiply(lr_arr, mhat),
                            mx::add(mx::sqrt(vhat), eps_arr)));
            break;
        }
        case 3: { // AdamW (decoupled weight decay)
            opt->m_bufs[i] = mx::add(mx::multiply(beta1_arr, opt->m_bufs[i]),
                                      mx::multiply(one_m_beta1, g));
            opt->v_bufs[i] = mx::add(mx::multiply(beta2_arr, opt->v_bufs[i]),
                                      mx::multiply(one_m_beta2, mx::square(g)));
            auto mhat = mx::divide(opt->m_bufs[i], bc1_arr);
            auto vhat = mx::divide(opt->v_bufs[i], bc2_arr);
            t->data = mx::subtract(t->data,
                mx::divide(mx::multiply(lr_arr, mhat),
                            mx::add(mx::sqrt(vhat), eps_arr)));
            t->data = mx::subtract(t->data,
                mx::multiply(scalar_like(lr * opt->weight_decay, t->data), t->data));
            break;
        }
        default:
            break;
        }
    }
    prof_optimizer_math_ms_mlx += _wall_ms_mlx() - tm0;

    // Eval all updated params
    std::vector<mx::array> to_eval;
    for (int i_ = 0; i_ < param_count(); i_++) {
        to_eval.push_back(((Tensor*)param_tensor(i_))->data);
    }
    mx::eval(to_eval);

    // Reset tape
    tape_reset();
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* p_tensor = (Tensor*)param_tensor(i_);
        p_tensor->tape_idx = -1;
        p_tensor->has_grad = false;
        tape_append(OP_CONST, p_tensor, nullptr, nullptr, 0);
    }
    prof_optimizer_ms_mlx += _wall_ms_mlx() - t0_opt;
    prof_epochs_mlx++;
}

/* Internal: clip grads for params matching prefix (empty prefix = all). */
static void clip_grad_value_filtered(const std::string& prefix, double max_val) {
    for (int i = 0; i < param_count(); i++) {
        std::string p_name = param_name(i);
        auto* p_tensor = (Tensor*)param_tensor(i);
        if (!prefix.empty() && p_name.rfind(prefix, 0) != 0) continue;
        if (p_tensor->has_grad) {
            auto lo = scalar_like(-max_val, p_tensor->grad);
            auto hi = scalar_like(max_val, p_tensor->grad);
            p_tensor->grad = mx::clip(p_tensor->grad, lo, hi);
        }
    }
}

static double clip_grad_norm_filtered(const std::string& prefix, double max_norm) {
    /* Compute the squared-grad sum per-param in the param's own dtype, then
       reduce to a double on the host. Avoids mixing dtypes in a single
       running `total` array (param dtypes may differ across the registry). */
    double sumsq = 0.0;
    for (int i = 0; i < param_count(); i++) {
        std::string p_name = param_name(i);
        auto* p_tensor = (Tensor*)param_tensor(i);
        if (!prefix.empty() && p_name.rfind(prefix, 0) != 0) continue;
        if (p_tensor->has_grad) {
            auto s = mx::sum(mx::square(p_tensor->grad));
            mx::eval(s);
            if (s.dtype() == mx::float64) sumsq += s.item<double>();
            else sumsq += (double)s.item<float>();
        }
    }
    double norm = std::sqrt(sumsq);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count(); i++) {
            std::string p_name = param_name(i);
            auto* p_tensor = (Tensor*)param_tensor(i);
            if (!prefix.empty() && p_name.rfind(prefix, 0) != 0) continue;
            if (p_tensor->has_grad) {
                p_tensor->grad = mx::multiply(p_tensor->grad,
                    scalar_like(scale, p_tensor->grad));
            }
        }
    }
    return norm;
}

void optimizer_clip_grad_value(double max_val) {
    clip_grad_value_filtered("", max_val);
}

double optimizer_clip_grad_norm(double max_norm) {
    return clip_grad_norm_filtered("", max_norm);
}

/* Polyak soft update: mirror of the tape/torch implementation. */
int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
    if (!online_scope || !target_scope) return 0;
    std::string on_s(online_scope), tg_s(target_scope);
    int blended = 0;
    for (int i = 0; i < param_count(); i++) {
        std::string on_name = param_name(i);
        if (on_name.rfind(on_s, 0) != 0) continue;
        std::string tgt_name = tg_s + on_name.substr(on_s.size());
        for (int j = 0; j < param_count(); j++) {
            if (std::string(param_name(j)) != tgt_name) continue;
            auto* on_t = (Tensor*)param_tensor(i);
            auto* tg_t = (Tensor*)param_tensor(j);
            if (on_t->data.shape() != tg_t->data.shape()) break;
            /* Build tau scalars matching the target dtype each iteration —
               cheap, and avoids dtype-mix UB when target params span dtypes. */
            auto tau_arr        = scalar_like(tau,       tg_t->data);
            auto one_minus_tau  = scalar_like(1.0 - tau, tg_t->data);
            tg_t->data = mx::add(
                mx::multiply(one_minus_tau, tg_t->data),
                mx::multiply(tau_arr, on_t->data));
            mx::eval(tg_t->data);
            blended++;
            break;
        }
    }
    return blended;
}

/* ================================================================
   Optimizer buffer accessors (for serialization)
   ================================================================ */

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return param_count();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    auto opt = (Optimizer*)h;
    if (idx >= (int)opt->m_bufs.size()) {
        int n = ((Tensor*)param_tensor(idx))->data.size();
        memset(out, 0, n * sizeof(double));
        return;
    }
    mx::eval(opt->m_bufs[idx]);
    auto& arr = opt->m_bufs[idx];
    mx_to_doubles(arr, out);
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    auto opt = (Optimizer*)h;
    if (idx >= (int)opt->v_bufs.size()) {
        int n = ((Tensor*)param_tensor(idx))->data.size();
        memset(out, 0, n * sizeof(double));
        return;
    }
    mx::eval(opt->v_bufs[idx]);
    auto& arr = opt->v_bufs[idx];
    mx_to_doubles(arr, out);
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    auto opt = (Optimizer*)h;
    // Ensure buffers exist
    int np = param_count();
    if ((int)opt->m_bufs.size() != np) {
        opt->m_bufs.clear();
        opt->v_bufs.clear();
        for (int i_ = 0; i_ < param_count(); i_++) {
            auto* p_tensor = (Tensor*)param_tensor(i_);
            opt->m_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
            opt->v_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
        }
    }
    auto t = (Tensor*)param_tensor(idx);
    opt->m_bufs[idx] = mx_array_from_doubles(data, t->data.shape(), t->data.dtype());
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    auto opt = (Optimizer*)h;
    int np = param_count();
    if ((int)opt->v_bufs.size() != np) {
        opt->m_bufs.clear();
        opt->v_bufs.clear();
        for (int i_ = 0; i_ < param_count(); i_++) {
            auto* p_tensor = (Tensor*)param_tensor(i_);
            opt->m_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
            opt->v_bufs.push_back(mx::zeros(p_tensor->data.shape(), p_tensor->data.dtype()));
        }
    }
    auto t = (Tensor*)param_tensor(idx);
    opt->v_bufs[idx] = mx_array_from_doubles(data, t->data.shape(), t->data.dtype());
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
    auto opt = (Optimizer*)h;
    out9[0] = (double)opt->type;
    out9[1] = opt->lr;
    out9[2] = opt->beta1;
    out9[3] = opt->beta2;
    out9[4] = opt->eps;
    out9[5] = opt->alpha;
    out9[6] = opt->weight_decay;
    out9[7] = opt->momentum;
    out9[8] = (double)opt->t;
}

void optimizer_set_meta(OptimizerHandle h, const double* in9) {
    auto opt = (Optimizer*)h;
    opt->type = (int)in9[0];
    opt->lr = in9[1];
    opt->beta1 = in9[2];
    opt->beta2 = in9[3];
    opt->eps = in9[4];
    opt->alpha = in9[5];
    opt->weight_decay = in9[6];
    opt->momentum = in9[7];
    opt->t = (int)in9[8];
}

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
void backend_epoch_begin(void) { /* no-op for MLX: profiling is backward+optimizer only */ }

void backend_profile_reset(void) {
    prof_backward_ms_mlx = prof_optimizer_ms_mlx = 0;
    prof_optimizer_math_ms_mlx = 0;
    prof_epochs_mlx = 0;
    prof_tape_appends_mlx = 0;
}

void backend_profile_report(void) {
    fprintf(stderr, "=== Profile Report (MLX backend) ===\n");
    fprintf(stderr, "  Epochs: %d\n", prof_epochs_mlx);
    fprintf(stderr, "  Params: %d tensors\n", param_count());
    fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n",
            prof_backward_ms_mlx, prof_epochs_mlx > 0 ? prof_backward_ms_mlx / prof_epochs_mlx : 0);
    fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n",
            prof_optimizer_ms_mlx, prof_epochs_mlx > 0 ? prof_optimizer_ms_mlx / prof_epochs_mlx : 0);
    fprintf(stderr, "    of which math: %.1fms total (%.1fms/epoch)\n",
            prof_optimizer_math_ms_mlx,
            prof_epochs_mlx > 0 ? prof_optimizer_math_ms_mlx / prof_epochs_mlx : 0);
    double total = prof_backward_ms_mlx + prof_optimizer_ms_mlx;
    fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n",
            total, prof_epochs_mlx > 0 ? total / prof_epochs_mlx : 0);
    fprintf(stderr, "  Forward tape_appends (grad-tracked ops): %ld total (%.0f/epoch)\n",
            prof_tape_appends_mlx,
            prof_epochs_mlx > 0 ? (double)prof_tape_appends_mlx / prof_epochs_mlx : 0);
}

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

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                         TensorHandle loss_ptr, double loss_val) {
    auto* o = (Optimizer*)opt;
    optimizer_zero_grad(opt);
    if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);
    if (clip_mode == 1) clip_grad_value_filtered(o->prefix, clip_val);
    else if (clip_mode == 2) clip_grad_norm_filtered(o->prefix, clip_val);
    optimizer_step(opt);
    return loss_val;
}
int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
    (void)dummy;
    auto* o = (Optimizer*)opt;
    if (clip_mode == 1) clip_grad_value_filtered(o->prefix, clip_val);
    else if (clip_mode == 2) clip_grad_norm_filtered(o->prefix, clip_val);
    optimizer_step(opt); optimizer_zero_grad(opt);
    return 0;
}
int tensor_live_count(int dummy) { (void)dummy; return (int)all_tensors.size(); }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_mlx_peak_live; }
/* dropout_random_seed lives in shared_utils.c. */

} // extern "C"



/* ---- Unified dtag-dispatch create/cast entry points ----
   One symbol per shape, dtag-keyed, superseding the per-dtype
   *_f32_streamed / *_f64_streamed wrappers above. mlx storage is
   f32/f64 only (Metal has no F64 on GPU, no bf16/f16/int storage), so
   dtag 0 → f32, 1 → f64, and the inference dtags 2-9 are rejected —
   symmetric with tape's abort. The Idris `Compatible` gate already
   prevents these dtags reaching mlx; the abort is a defence-in-depth
   backstop naming the symbol. */
[[noreturn]] static TensorHandle mlx_dtype_unsupported(const char* sym, int dtag) {
    fprintf(stderr,
        "[mlx backend] %s called with dtag=%d. mlx stores f32/f64 only "
        "(Metal has no bf16/f16/int storage). Bind your code to F32/F64 "
        "on mlx, or build with BACKEND=torch.\n", sym, dtag);
    abort();
}

TensorHandle tensor_create_scalar_streamed(double value, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_scalar_f32_mlx_streamed(value, requires_grad, stream_tag);
        case 15: return tensor_create_scalar_f64_mlx_streamed(value, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_scalar_streamed", dtag);
    }
}
TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_f32_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
        case 15: return tensor_create_f64_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_streamed", dtag);
    }
}
TensorHandle tensor_create_1d_streamed(int n, double* data, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_1d_f32_mlx_streamed(n, data, requires_grad, stream_tag);
        case 15: return tensor_create_1d_f64_mlx_streamed(n, data, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_1d_streamed", dtag);
    }
}
TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_2d_f32_mlx_streamed(rows, cols, data, requires_grad, stream_tag);
        case 15: return tensor_create_2d_f64_mlx_streamed(rows, cols, data, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_2d_streamed", dtag);
    }
}
TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_1d_f32_mlx_streamed(n, data, stream_tag);
        case 15: return tensor_create_param_1d_f64_mlx_streamed(n, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_1d_streamed", dtag);
    }
}
TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_2d_f32_mlx_streamed(rows, cols, data, stream_tag);
        case 15: return tensor_create_param_2d_f64_mlx_streamed(rows, cols, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_2d_streamed", dtag);
    }
}
TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_3d_f32_mlx_streamed(d0, d1, d2, data, stream_tag);
        case 15: return tensor_create_param_3d_f64_mlx_streamed(d0, d1, d2, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_3d_streamed", dtag);
    }
}
TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_4d_f32_mlx_streamed(d0, d1, d2, d3, data, stream_tag);
        case 15: return tensor_create_param_4d_f64_mlx_streamed(d0, d1, d2, d3, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_4d_streamed", dtag);
    }
}
TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_1d_f32_mlx_streamed(n, data, stream_tag);
        case 15: return tensor_create_state_1d_f64_mlx_streamed(n, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_state_1d_streamed", dtag);
    }
}
TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_2d_f32_mlx_streamed(rows, cols, data, stream_tag);
        case 15: return tensor_create_state_2d_f64_mlx_streamed(rows, cols, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_state_2d_streamed", dtag);
    }
}
TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_cast_dtype_f32_mlx_streamed(src, stream_tag);
        case 15: return tensor_cast_dtype_f64_mlx_streamed(src, stream_tag);
        default: mlx_dtype_unsupported("tensor_cast_dtype_streamed", dtag);
    }
}

/* ---------- Shared training port adapter ----------
 *
 * Provides the per-tensor accessors that shared/training/param_registry.c
 * uses to talk to mlx (numel / has_grad / grad read+write / zero / data
 * read+write / bulk doubles+int64 loaders) plus tensor_retain/release
 * for the refcount lifecycle the shared registry keeps anchored. Other
 * port slots stay nullptr until mlx opts into the matching shared TUs
 * (optimizer_*, dtag-streamed creators, ffi_shims). */

#include "shared/training/port.h"

static int mlx_port_tensor_numel(void* h) {
    return (int)((Tensor*)h)->data.size();
}

static int mlx_port_tensor_requires_grad(void* h) {
    return ((Tensor*)h)->requires_grad ? 1 : 0;
}

static int mlx_port_tensor_has_grad(void* h) {
    return ((Tensor*)h)->has_grad ? 1 : 0;
}

static double mlx_port_grad_read(void* h, int i) {
    auto* t = (Tensor*)h;
    if (!t->has_grad) return 0.0;
    /* mx::vjp may return non-contiguous arrays; force contiguous read. */
    auto contig = mx::contiguous(t->grad);
    mx::eval(contig);
    return mx_read_double(contig, i);
}

static void mlx_port_grad_write(void* h, int i, double v) {
    auto* t = (Tensor*)h;
    if (!t->has_grad) return;
    /* Realize grad host-side, mutate element, push back. mlx arrays are
       immutable, so writing one element requires rebuilding via
       mx_array_from_doubles. Param-registry callers hit this rarely
       (only param_subtract_delta / param_grad_item_and_zero on scalar
       params), so per-call allocate is acceptable. */
    auto contig = mx::contiguous(t->grad);
    mx::eval(contig);
    int n = (int)contig.size();
    double* buf = (double*)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) buf[k] = mx_read_double(contig, k);
    buf[i] = v;
    t->grad = mx_array_from_doubles(buf, t->grad.shape(), t->grad.dtype());
    free(buf);
}

static void mlx_port_zero_grad(void* h) {
    auto* t = (Tensor*)h;
    if (t->has_grad) {
        t->grad = mx::zeros(t->data.shape(), t->data.dtype());
    }
}

static double mlx_port_data_read(void* h, int i) {
    auto* t = (Tensor*)h;
    auto contig = mx::contiguous(t->data);
    mx::eval(contig);
    return mx_read_double(contig, i);
}

static void mlx_port_data_write(void* h, int i, double v) {
    auto* t = (Tensor*)h;
    auto contig = mx::contiguous(t->data);
    mx::eval(contig);
    int n = (int)contig.size();
    double* buf = (double*)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) buf[k] = mx_read_double(contig, k);
    buf[i] = v;
    t->data = mx_array_from_doubles(buf, t->data.shape(), t->data.dtype());
    free(buf);
}

static void mlx_port_load_doubles(void* h, const double* src, int n) {
    auto* t = (Tensor*)h;
    (void)n;  /* shape already determines size; caller validates against numel */
    t->data = mx_array_from_doubles(src, t->data.shape(), t->data.dtype());
}

static void mlx_port_load_int64(void* h, const int64_t* src, int n) {
    auto* t = (Tensor*)h;
    /* No native I64 storage on mlx — pivot through double. Matches the
       lossy I64 behavior the previous in-file param_load_data_int64
       documented (values above 2^53 lose precision). */
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) tmp[k] = (double)src[k];
    t->data = mx_array_from_doubles(tmp, t->data.shape(), t->data.dtype());
    free(tmp);
}

const BackendPort g_active_port = {
    /* Tensor introspection + per-element + bulk grad/data ops. */
    .tensor_numel              = mlx_port_tensor_numel,
    .tensor_requires_grad      = mlx_port_tensor_requires_grad,
    .tensor_has_grad           = mlx_port_tensor_has_grad,
    .data_read                 = mlx_port_data_read,
    .data_write                = mlx_port_data_write,
    .grad_read                 = mlx_port_grad_read,
    .grad_write                = mlx_port_grad_write,
    .zero_grad                 = mlx_port_zero_grad,
    .load_doubles              = mlx_port_load_doubles,
    .load_int64                = mlx_port_load_int64,
    /* Remaining slots wait on mlx joining the corresponding shared TUs.
       Order matches port.h's struct declaration order (C++ ISO-required
       for designated init). */
    .backward                  = nullptr,
    .optimizer_create_sgd      = nullptr,
    .optimizer_create_rmsprop  = nullptr,
    .optimizer_create_adam     = nullptr,
    .optimizer_create_adam_group = nullptr,
    .optimizer_create_adamw    = nullptr,
    .optimizer_free            = nullptr,
    .optimizer_set_lr          = nullptr,
    .optimizer_set_param_lr    = nullptr,
    .optimizer_step            = nullptr,
    .optimizer_buf_count       = nullptr,
    .optimizer_get_m           = nullptr,
    .optimizer_get_v           = nullptr,
    .optimizer_set_m           = nullptr,
    .optimizer_set_v           = nullptr,
    .optimizer_get_meta        = nullptr,
    .optimizer_set_meta        = nullptr,
    .wall_ms                   = nullptr,
    .create_scalar             = nullptr,
    .create                    = nullptr,
    .create_1d                 = nullptr,
    .create_2d                 = nullptr,
    .create_param_1d           = nullptr,
    .create_param_2d           = nullptr,
    .create_param_3d           = nullptr,
    .create_param_4d           = nullptr,
    .create_state_1d           = nullptr,
    .create_state_2d           = nullptr,
    .cast_dtype                = nullptr,
};

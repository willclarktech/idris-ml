/* Dtype dispatch — torch.
 *
 * Holds the dtype-routing layer between the public FFI (one symbol per
 * shape, dtag-keyed) and libtorch's typed creation API:
 *
 *   - `idrisml_is_floating_st` — predicate that gates `requires_grad`
 *     activation (libtorch throws on non-floating).
 *   - `st_for_dtag` — the runtime `dtag → torch::ScalarType` lookup.
 *     Aborts on unknown dtag so misuse fails loud, not silent-cast.
 *   - `make_param_leaf` — cast-before-requires_grad parameter builder.
 *     Cast-after-grad would yield a non-leaf whose `.grad` never
 *     populates (silent training freeze on F32). See gotchas.md.
 *   - `create_*_dt` (4 shapes) — generic dtype-parameterised builders
 *     for inference dtypes (BF16, F16, Int, Bool).
 *   - F32/F64 explicit-suffix variants (`tensor_create_*_{f32,f64}`) —
 *     FFI-named per-dtype entry points; F64s alias the base creators,
 *     F32s build at fp64 then cast.
 *   - `torch_create_*_dtag` family + `torch_cast_dtype_dtag` — the
 *     per-shape dtag dispatchers; wired into the shared training port
 *     via adapter.cpp.
 */
#include "dtype_dispatch.h"
#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* Base creators that the F32/F64 wrappers + dtag dispatchers delegate
   into; defined in backend_torch.cpp. */
extern "C" TensorHandle tensor_create_scalar_f32(double v, int rg);
extern "C" TensorHandle tensor_create_scalar_f64(double v, int rg);
extern "C" TensorHandle tensor_create_f32(double* data, int* shape, int rank, int rg);
extern "C" TensorHandle tensor_create_f64(double* data, int* shape, int rank, int rg);
extern "C" TensorHandle tensor_create_2d(int rows, int cols, double* data, int rg);
extern "C" TensorHandle tensor_create_param_1d(int n, double* data);
extern "C" TensorHandle tensor_create_param_2d(int rows, int cols, double* data);
extern "C" TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data);
extern "C" TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data);
extern "C" TensorHandle tensor_create_state_1d(int n, double* data);
extern "C" TensorHandle tensor_create_state_2d(int rows, int cols, double* data);

/* Process-wide target device pin set at dylib load from TORCH_DEVICE
   (see mps_init.cpp). The streamed-path creators consult this so that
   `BACKEND=torch TORCH_DEVICE=mps` actually lands layer params + inputs
   on MPS instead of silently staying on CPU. Default at::kCPU keeps the
   no-env path identical to the historical behaviour. */
extern c10::Device g_torch_target_device;

/* Migrate `t` to the active target device IF it differs from `t.device()`.
   Stays in-place when the device matches (the common case for CPU builds);
   issues a single .to(device) op when migrating. */
static inline at::Tensor torch_migrate_to_target(at::Tensor t) {
    return t.device() == g_torch_target_device ? t : t.to(g_torch_target_device);
}

/* ---- Floating-dtype predicate ----
   torch rejects requires_grad on int/bool tensors; we silently skip
   `requires_grad_(true)` rather than abort, since registerParam-only
   (no-grad) registration is a legitimate use case for inference dtypes. */
bool idrisml_is_floating_st(torch::ScalarType dt) {
    return dt == torch::kFloat32 || dt == torch::kFloat64 ||
           dt == torch::kBFloat16 || dt == torch::kHalf;
}

/* ---- Inference-only dtype scaffolding (BF16, F16, Int, Bool) ----
   Generic dtype-parameterised create/cast over the lean non-grad set
   (scalar/create/1d/2d/cast). requires_grad is honored only for floating
   dtypes; torch rejects autograd on integer/bool. */
static TensorHandle create_scalar_dt(double v, int rg, torch::ScalarType dt) {
    auto t = torch::tensor(v, torch::dtype(dt));
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}
static TensorHandle create_nd_dt(double* data, int* shape, int rank, int rg, torch::ScalarType dt) {
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; i++) dims[i] = shape[i];
    auto t = torch::from_blob(data, dims, torch::kFloat64).clone();
    if (dt != torch::kFloat64) t = t.to(dt);
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}
static TensorHandle create_1d_dt(int n, double* d, int rg, torch::ScalarType dt) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    free(d);
    if (dt != torch::kFloat64) t = t.to(dt);
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor(std::move(t));
}
static TensorHandle create_2d_dt(int rows, int cols, double* d, int rg, torch::ScalarType dt) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(d);
    if (dt != torch::kFloat64) t = t.to(dt);
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

/* ---- Cast helper used by F32 state creators + the dtag dispatchers ---- */
TensorHandle torch_cast_to(TensorHandle h, torch::ScalarType dt) {
    auto t = *to_tensor(h);
    return from_tensor_persistent(t.dtype() == dt ? t : t.to(dt));
}

/* ---- Cast-and-move-before-requires_grad parameter builder ----
   Cast-and-move-before-requires_grad is load-bearing: `.to(dt)` or
   `.to(device)` applied to an already-requires_grad tensor yields a
   NON-LEAF (the ToCopy output), whose .grad never populates during
   backward — the optimizer then reads a zero gradient and silently
   no-ops, freezing training at the init loss.

   Cast + device migration are combined into a single `.to(opts)` call so
   the leaf transitions happen in one step. When dt + device both match
   the F64-CPU source, we skip the .to() entirely (no-op fast path). */
TensorHandle make_param_leaf(double* data, c10::IntArrayRef dims, torch::ScalarType dt) {
    auto t = torch::from_blob(data, dims, torch::kFloat64).clone();
    bool need_cast   = dt != torch::kFloat64;
    bool need_move   = g_torch_target_device != at::kCPU;
    if (need_cast || need_move) {
        auto opts = torch::TensorOptions().dtype(dt).device(g_torch_target_device);
        t = t.to(opts);
    }
    t.requires_grad_(true);
    // A parameter must be an autograd leaf or its .grad never populates and
    // the optimizer silently no-ops (frozen training, no error). This fires
    // immediately at the construction site if a future change reorders the
    // cast/move after requires_grad_ on any backend build. See gotchas.md
    // "A parameter must be cast/moved before requires_grad_".
    TORCH_CHECK(t.is_leaf(),
        "parameter tensor is not an autograd leaf: cast/move (.to(dtype/device)) "
        "must precede requires_grad_, otherwise .grad never populates and the "
        "optimizer silently freezes training");
    return from_tensor_persistent(std::move(t));
}

/* ---- F64 explicit-suffix wrappers ----
   F64 creators land at fp64 on CPU then migrate to g_torch_target_device
   if it differs. `tensor_create_1d_f64` is the unsuffixed-1d's
   replacement (the legacy unsuffixed symbol was retired in f7a690b); 2d
   still has a CPU-only unsuffixed entry which the no-grad/F64 path on
   tape + cpu-torch reuses, so the 2d F64 wrapper migrates after that. */
extern "C" TensorHandle tensor_create_1d_f64(int n, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    free(d);
    t = torch_migrate_to_target(std::move(t));
    if (rg) t.requires_grad_(true);
    return from_tensor(std::move(t));
}
extern "C" TensorHandle tensor_create_2d_f64(int rows, int cols, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(d);
    t = torch_migrate_to_target(std::move(t));
    if (rg) t.requires_grad_(true);
    return from_tensor(std::move(t));
}
/* F64 param/state wrappers go through make_param_leaf / state-create + migrate. */
extern "C" TensorHandle tensor_create_param_1d_f64(int n, double* d)                               { return make_param_leaf(d, {(int64_t)n}, torch::kFloat64); }
extern "C" TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* d)                  { return make_param_leaf(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64); }
extern "C" TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* d)              { return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat64); }
extern "C" TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* d)      { return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat64); }
extern "C" TensorHandle tensor_create_state_1d_f64(int n, double* d) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    return from_tensor_persistent(torch_migrate_to_target(std::move(t)));
}
extern "C" TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* d) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    return from_tensor_persistent(torch_migrate_to_target(std::move(t)));
}

/* ---- F32 explicit-suffix wrappers ----
   Build at fp64 then cast + migrate via a single torch::TensorOptions.to(opts)
   call (kept atomic for the make_param_leaf leaf-discipline; non-param paths
   set requires_grad after migration but don't need a leaf so the ordering is
   only constrained by performance). */
extern "C" TensorHandle tensor_create_1d_f32(int n, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    free(d);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(g_torch_target_device);
    t = t.to(opts);
    if (rg) t.requires_grad_(true);   // cast/move-before-grad: keep the F32 tensor a leaf
    return from_tensor(std::move(t));
}
extern "C" TensorHandle tensor_create_2d_f32(int rows, int cols, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(d);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(g_torch_target_device);
    t = t.to(opts);
    if (rg) t.requires_grad_(true);   // cast/move-before-grad: keep the F32 tensor a leaf
    return from_tensor(std::move(t));
}
extern "C" TensorHandle tensor_create_param_1d_f32(int n, double* d) {
    return make_param_leaf(d, {(int64_t)n}, torch::kFloat32);
}
extern "C" TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* d) {
    return make_param_leaf(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat32);
}
extern "C" TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* d) {
    return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat32);
}
extern "C" TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* d) {
    return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat32);
}
extern "C" TensorHandle tensor_create_state_1d_f32(int n, double* d) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(g_torch_target_device);
    return from_tensor_persistent(t.to(opts));
}
extern "C" TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* d) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(g_torch_target_device);
    return from_tensor_persistent(t.to(opts));
}

/* ---- Dtag → torch::ScalarType ----
   The kind-major dtag layout the Idris-side RuntimeDType encodes;
   invalid dtags abort (silent miscast is worse than a loud fail). */
torch::ScalarType st_for_dtag(int dtag) {
    switch (dtag) {
        case 1:  return torch::kBool;       /* Bool */
        case 4:  return torch::kByte;       /* U8 */
        case 8:  return torch::kChar;       /* I8 */
        case 9:  return torch::kShort;      /* I16 */
        case 10: return torch::kInt;        /* I32 */
        case 11: return torch::kLong;       /* I64 */
        case 13: return torch::kHalf;       /* F16 */
        case 14: return torch::kFloat32;    /* F32 */
        case 15: return torch::kFloat64;    /* F64 */
        case 17: return torch::kBFloat16;   /* BF16 */
        default:
            std::fprintf(stderr,
                "invalid dtag %d: expected one of {1=Bool, 4=U8, 8-11=I8/I16/I32/I64, "
                "13-15=F16/F32/F64, 17=BF16}\n", dtag);
            std::abort();
    }
}

/* ---- Per-shape dtag dispatchers (wired into the shared port) ----
   F32/F64 route to the dedicated dtype creators (byte-identical with the
   previous in-file path); other dtags route through the generic
   create_*_dt / make_param_leaf / torch_cast_to path. */
TensorHandle torch_create_scalar_dtag(double v, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_scalar_f32(v, rg);
        case 15: return tensor_create_scalar_f64(v, rg);
        default: return create_scalar_dt(v, rg, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_dtag(double* data, int* shape, int rank, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_f32(data, shape, rank, rg);
        case 15: return tensor_create_f64(data, shape, rank, rg);
        default: return create_nd_dt(data, shape, rank, rg, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_1d_dtag(int n, double* data, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_1d_f32(n, data, rg);
        case 15: return tensor_create_1d_f64(n, data, rg);
        default: return create_1d_dt(n, data, rg, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_2d_dtag(int rows, int cols, double* data, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_2d_f32(rows, cols, data, rg);
        case 15: return tensor_create_2d_f64(rows, cols, data, rg);
        default: return create_2d_dt(rows, cols, data, rg, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_param_1d_dtag(int n, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_1d_f32(n, data);
        case 15: return tensor_create_param_1d_f64(n, data);
        default: return make_param_leaf(data, {(int64_t)n}, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_param_2d_dtag(int rows, int cols, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_2d_f32(rows, cols, data);
        case 15: return tensor_create_param_2d_f64(rows, cols, data);
        default: return make_param_leaf(data, {(int64_t)rows, (int64_t)cols}, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_3d_f32(d0, d1, d2, data);
        case 15: return tensor_create_param_3d_f64(d0, d1, d2, data);
        default: return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_4d_f32(d0, d1, d2, d3, data);
        case 15: return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
        default: return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, st_for_dtag(dtag));
    }
}
TensorHandle torch_create_state_1d_dtag(int n, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_1d_f32(n, data);
        case 15: return tensor_create_state_1d_f64(n, data);
        default: return torch_cast_to(tensor_create_state_1d(n, data), st_for_dtag(dtag));
    }
}
TensorHandle torch_create_state_2d_dtag(int rows, int cols, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_2d_f32(rows, cols, data);
        case 15: return tensor_create_state_2d_f64(rows, cols, data);
        default: return torch_cast_to(tensor_create_state_2d(rows, cols, data), st_for_dtag(dtag));
    }
}
TensorHandle torch_cast_dtype_dtag(TensorHandle src, int dtag) {
    return from_tensor(to_tensor(src)->to(st_for_dtag(dtag)));
}

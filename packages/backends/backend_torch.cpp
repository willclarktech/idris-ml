#include "backend.h"
#include "backend_torch/tensor.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cstring>
#include <string>
#include <vector>
#include <unordered_set>
#include <sys/resource.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* ---------- MPS eager-init ----------
 * libtorch lazily initializes its MPS allocator + Metal command queue on
 * the first tensor that touches MPS. In multi-backend builds where cross-
 * backend tensor transfers run early (test-multi's Transfer suite), that
 * lazy init races macOS work-queue threads (MTLDevice setup, MPS
 * allocator pool ramp-up) and sporadically aborts the process with
 * SIGSEGV inside libtorch's `at::native::mps::*` paths. We saw this
 * empirically: re-ordering tests so an intra-torch CPU→MPS migration
 * runs FIRST dropped the crash rate from 100% to ~3%. Forcing the
 * init at dylib-load time (here) closes the window entirely — by the
 * time any Idris-side code calls `tensor_to_device_torch(h, "mps")`,
 * the MPS subsystem is already warm.
 *
 * Cost: one MPS tensor alloc+dealloc at process start (microseconds).
 * Skipped if torch wasn't built with MPS support or if no MPS device
 * is available on this host (Linux CI, non-Apple-Silicon Mac). */
__attribute__((constructor))
static void torch_mps_eager_init(void) {
    if (!at::hasMPS()) return;
    try {
        auto opts = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::Device(at::DeviceType::MPS));
        auto warm = torch::zeros({1}, opts);
        // Touch the data once to force the Metal command buffer to drain
        // (synchronize). The `.cpu()` round-trip is what
        // tensor_to_doubles_torch does at runtime — same code path.
        (void)warm.cpu();
        // `warm` falls out of scope; its storage refcount hits zero
        // and libtorch returns the MPS buffer to the allocator pool.
    } catch (...) {
        // First-touch failures (paravirt-MPS quirks on Tart VMs etc.)
        // shouldn't prevent dylib load. Subsequent Idris-side MPS use
        // will surface the real error.
    }
}

/* Profiling counters (prof_backward_ms / prof_optimizer_ms /
   prof_optimizer_math_ms / prof_epochs) + _wall_ms_torch live in
   backend_torch/training/profiling.{h,cpp}. */
#include "backend_torch/training/profiling.h"

/* Intermediate tensor tracking (`intermediates`, `all_pairs`,
   `tracking_enabled`, `g_torch_peak_live_intermediates`) lives in
   backend_torch/training/intermediates.{h,cpp}. */
#include "backend_torch/training/intermediates.h"

/* ---------- Helpers ---------- */

// to_tensor lives inline in backend_torch/tensor.h (the modular tree
// header). from_tensor / from_tensor_persistent are declared there too,
// but defined here so they remain co-located with the intermediates
// list they push into.
TensorHandle from_tensor(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    if (tracking_enabled_torch) {
        intermediates_torch.push_back(p);
        if ((long)intermediates_torch.size() > g_torch_peak_live_intermediates)
            g_torch_peak_live_intermediates = (long)intermediates_torch.size();
    }
    return static_cast<TensorHandle>(p);
}

// Persistent variant: not tracked for cleanup (survives optimizer_step)
TensorHandle from_tensor_persistent(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    return static_cast<TensorHandle>(p);
}
static TensorHandle make_param_leaf(double* data, c10::IntArrayRef dims, torch::ScalarType dt); // defined near the F32 param creators

/* ---------- Lifecycle ----------
   tensor_create_scalar* / tensor_create* / tensor_cast_dtype_* extracted
   to backend_torch/core/lifecycle/{create_scalar,create,cast}.cpp. */

/* tensor_clone / tensor_free / tensor_retain_handle / tensor_release_handle
   extracted to backend_torch/core/lifecycle/. The `freed_by_cleanup`
   set + free_intermediates() impl live in
   backend_torch/training/intermediates.cpp. */

/* ---------- Accessors ---------- */

/* tensor_item / tensor_item_1d / tensor_item_2d extracted to
   backend_torch/core/lifecycle/. tensor_item_1d / item_2d definitions
   that lived below are also gone. */

int tensor_numel(TensorHandle h) {
    return static_cast<int>(to_tensor(h)->numel());
}

int tensor_dim(TensorHandle h) {
    return static_cast<int>(to_tensor(h)->dim());
}

int tensor_size(TensorHandle h, int dim) {
    return static_cast<int>(to_tensor(h)->size(dim));
}

void tensor_to_doubles(TensorHandle h, double* out) {
    // .cpu() before .data_ptr<>() — readback to host memory needs the
    // tensor on CPU. F64 on MPS isn't supported at construction so the
    // .to(kFloat64) for an MPS source goes through .cpu() first.
    auto t = to_tensor(h)->cpu().to(torch::kFloat64).contiguous();
    std::memcpy(out, t.data_ptr<double>(), t.numel() * sizeof(double));
}

void tensor_to_floats(TensorHandle h, float* out) {
    auto t = to_tensor(h)->cpu().to(torch::kFloat32).contiguous();
    std::memcpy(out, t.data_ptr<float>(), t.numel() * sizeof(float));
}

const char* tensor_dtype_name(TensorHandle h) {
    switch (to_tensor(h)->scalar_type()) {
        case torch::kFloat32:  return "F32";
        case torch::kFloat64:  return "F64";
        case torch::kBFloat16: return "BF16";
        case torch::kHalf:     return "F16";
        case torch::kChar:     return "I8";
        case torch::kShort:    return "I16";
        case torch::kInt:      return "I32";
        case torch::kLong:     return "I64";
        case torch::kByte:     return "U8";
        case torch::kBool:     return "BOOL";
        default:               return "F64";
    }
}

/* ---------- Arithmetic ---------- */

/* tensor_add / tensor_sub / tensor_mul / tensor_div extracted to
   backend_torch/core/elementwise/{add,sub,mul,div}.cpp. */

/* tensor_neg / tensor_abs / tensor_exp / tensor_log / tensor_sqrt /
   tensor_pow extracted to backend_torch/core/elementwise/. */

/* tensor_sigmoid / tensor_tanh / tensor_softplus extracted to
   backend_torch/core/elementwise/. tensor_gelu / tensor_leaky_relu /
   tensor_silu live in backend_torch/nn/activation/. */

/* tensor_add_scalar / tensor_mul_scalar / tensor_clamp_min extracted
   to backend_torch/core/scalar/. */

/* Reduction ops live in backend_torch/linear/reduction/ */

/* Linear algebra (matmul, mv, mm, linear, linear_2d, dot, outer, bmm,
 * bmm_3x3, transpose_2d, transpose_last2, tile_2d) lives in
 * backend_torch/linear/linalg/. */

/* Softmax / log_softmax (incl. _2d / _3d variants) live in
 * backend_torch/nn/softmax/. */

/* Loss ops (bce_with_logits, cross_entropy, mse_loss) live in
 * backend_torch/nn/loss/. cosine_similarity lives in
 * backend_torch/nn/attention/. */

/* Norm ops (batch_norm, group_norm, dropout, layer_norm_2d) live in
 * backend_torch/nn/norm/. Attention ops (cross_attention, embedding,
 * cosine_similarity) live in backend_torch/nn/attention/. Recurrent
 * ops (gru_cell, lstm_cell, pair_helpers) live in
 * backend_torch/nn/recurrent/. */

/* Index ops (gather, scatter_add) live in backend_torch/linear/index/.
 * Sort ops (argsort, cumprod) live in backend_torch/linear/sort/.
 *
 * Conv / pool ops (conv1d/2d + circular + transpose + grouped +
 * max_pool1d/2d[_batched] + avg_pool1d/2d) live in
 * backend_torch/conv/. */

TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat64);
}

/* Shape ops live in backend_torch/linear/shape/.
 * Stack / cat live in backend_torch/linear/concat/. */

/* ---------- Autograd ---------- */

extern "C" void _dbg_dump_param_grads_if_enabled_torch(void);

void tensor_backward(TensorHandle h) {
    double t0 = _wall_ms_torch();
    to_tensor(h)->backward();
    prof_backward_ms_torch += _wall_ms_torch() - t0;
    /* Phase 1.5e diagnostic: dump per-param gradient L2 norms after backward.
       Implementation lives below the param_registry declaration. */
    _dbg_dump_param_grads_if_enabled_torch();
}

TensorHandle tensor_grad(TensorHandle h) {
    auto& g = to_tensor(h)->grad();
    if (!g.defined()) return nullptr;
    return from_tensor(g);
}

void tensor_zero_grad(TensorHandle h) {
    auto& t = *to_tensor(h);
    if (t.grad().defined()) {
        t.grad().zero_();
    }
}

int tensor_requires_grad(TensorHandle h) {
    return to_tensor(h)->requires_grad() ? 1 : 0;
}

TensorHandle tensor_detach(TensorHandle h) {
    return from_tensor(to_tensor(h)->detach());
}

TensorHandle tensor_with_grad(TensorHandle h) {
    auto t = to_tensor(h)->detach().clone();
    t.requires_grad_(true);
    return from_tensor(std::move(t));
}

static inline bool idrisml_is_floating_st(torch::ScalarType dt);  /* defined below */

void tensor_set_requires_grad(TensorHandle h, int requires_grad) {
    auto t = to_tensor(h);
    /* torch throws if you request grad on a non-floating tensor. Inference
       dtypes (int/bool) can be registered (e.g. via registerParam, for
       serialization) but can't carry gradients — silently leave grad off
       rather than abort. */
    if (requires_grad && !idrisml_is_floating_st(t->scalar_type())) return;
    t->requires_grad_(requires_grad != 0);
}

/* No-grad scope. Counter (not bool) so nested withNoGrad scopes
   nest correctly — only the outermost begin creates the guard,
   only the outermost end releases it. */
static thread_local int no_grad_depth = 0;
static thread_local std::unique_ptr<torch::NoGradGuard> no_grad_guard;

void tensor_no_grad_begin(void) {
    if (no_grad_depth == 0) {
        no_grad_guard = std::make_unique<torch::NoGradGuard>();
    }
    no_grad_depth++;
}

void tensor_no_grad_end(void) {
    if (no_grad_depth > 0) {
        no_grad_depth--;
        if (no_grad_depth == 0) {
            no_grad_guard.reset();
        }
    }
}
/* No buffer ceiling on torch; per-epoch generation free is a no-op. */
void tensor_epoch_begin(void) {}
void tensor_epoch_end(void) {}

/* ---------- Device ---------- */

/* EAFP availability gate: a device-pin to absent/invalid hardware
 * (e.g. "cuda:1" on a 1-GPU box, or MPS on a non-Apple host) makes
 * libtorch's `.to()` throw a c10::Error. Unguarded, that exception
 * crosses the C->Chez FFI boundary and becomes std::terminate/SIGABRT.
 * Catch it here and return a NULL handle; the Idris side lifts NULL ->
 * Left DeviceError. This is the one source of truth for availability —
 * no separate is_available probe to drift. All torch device-pinning
 * (primCreateFromHost, primIntraMigrate, primCreate's post-create
 * migration) routes through here, so this single guard covers them. */
TensorHandle tensor_to_device(TensorHandle h, const char* device) {
    try {
        return from_tensor(to_tensor(h)->to(std::string(device)));
    } catch (const std::exception& e) {
        fprintf(stderr, "[torch] tensor_to_device(%s) failed: %s\n",
                device, e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "[torch] tensor_to_device(%s) failed: unknown\n",
                device);
        return nullptr;
    }
}

static thread_local std::string device_str;

const char* tensor_device(TensorHandle h) {
    auto d = to_tensor(h)->device();
    device_str = d.str();
    return device_str.c_str();
}

/* _dbg_dump_lstm_traj_if_enabled_torch + _dbg_dump_param_grads_if_enabled_torch
   live in backend_torch/training/diagnostics.cpp. */
extern "C" void _dbg_dump_lstm_traj_if_enabled_torch(void);
extern "C" void _dbg_dump_param_grads_if_enabled_torch(void);

/* The param-registry surface (param_register / param_count / param_name
   / param_tensor / param_grad_item* / param_zero_all_grads /
   param_subtract_delta / param_load_data / param_load_data_int64) lives
   in shared/training/param_registry.c and is opted in via the
   SHARED_BACKENDS_param_registry Makefile list. Torch supplies the
   per-tensor accessors (tensor_numel / tensor_has_grad / grad_read /
   grad_write / zero_grad / data_read / data_write / load_doubles /
   load_int64) through its port adapter at the bottom of this file. */

// Byte-exact I64 readout — bypasses the double pivot so values
// above 2^53 survive. The `.to(kInt64)` is a no-op when the source
// is already I64; if a caller asks for int64 readout on a non-I64
// tensor we still narrow through torch's standard truncating cast
// (matches the implicit cast in `tensor_to_doubles` for the same
// case). `.cpu()` mirrors the device handling in `tensor_to_doubles`.
void tensor_to_int64(TensorHandle h, int64_t* out) {
    auto t = to_tensor(h)->cpu().to(torch::kInt64).contiguous();
    std::memcpy(out, t.data_ptr<int64_t>(), t.numel() * sizeof(int64_t));
}

/* tensor_subtract_scalar_inplace lives in
   backend_torch/training/ntm_specific.cpp. */

/* ---------- Convenience ---------- */

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    free(data);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(data);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

/* tensor_alloc_doubles / tensor_free_doubles / tensor_read_double /
 * tensor_ptr_array_alloc live in shared_utils.c. */

/* tensor_stack_from_array / tensor_cat_from_array live in
 * backend_torch/linear/concat/{stack,cat}.cpp. */


/* Forward decl — st_for_dtag is defined further down with the unified
   create/cast block; tensor_one_hot below calls it. */
static torch::ScalarType st_for_dtag(int dtag);

TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
    int total = n_tokens * vocab_size;
    // Build the 0/1 pattern in F64, then cast to the requested dtype so the
    // result honestly matches the Idris `dt` (0/1 is exact in every dtype —
    // float or int — so the cast is lossless). An F32 model gets a real F32
    // one-hot, an F64 model a real F64 one — no silent dtype divergence.
    auto t = torch::zeros({(int64_t)total}, torch::kFloat64);
    auto acc = t.accessor<double, 1>();
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            acc[i * vocab_size + tok] = 1.0;
    }
    /* Delegate to st_for_dtag for the kind-major dtag layout; invalid
       dtags abort there. F64 is the build dtype above, so skip the cast
       only when the requested output dtype is already F64. */
    torch::ScalarType st = st_for_dtag(dtag);
    if (st != torch::kFloat64) t = t.to(st);
    return from_tensor(std::move(t));
}

TensorHandle tensor_batch(TensorHandle* handles, int count) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(handles[i]);
    return from_tensor(torch::stack(vec));
}

TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
    auto tensors = to_tensor(h)->unbind(0);
    *out_count = (int)tensors.size();
    auto* arr = (TensorHandle*)malloc(*out_count * sizeof(TensorHandle));
    for (int i = 0; i < *out_count; i++)
        arr[i] = from_tensor(tensors[i].contiguous());
    return arr;
}

/* ---------- Tensor-level parameter creation ---------- */

TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    return make_param_leaf(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64);
}

TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat64);
}

TensorHandle tensor_create_param_1d(int n, double* data) {
    return make_param_leaf(data, {(int64_t)n}, torch::kFloat64);
}

TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

/* ================================================================
   Per-dtype creation variants
   --------------------------------------------------------------
   Torch supports F32 and F64 natively (both first-class). The _f64
   variants are functionally identical to the existing unsuffixed
   creators; the _f32 variants build with kFloat32 and cast the
   double-typed input buffer down.
   ================================================================ */

static TensorHandle torch_cast_to(TensorHandle h, torch::ScalarType dt) {
    auto t = *to_tensor(h);
    return from_tensor_persistent(t.dtype() == dt ? t : t.to(dt));
}

/* Build a persistent grad-tracking leaf at dtype dt from a fp64 host buffer.
   Cast-before-requires_grad is load-bearing: `.to(dt)` applied to an
   already-requires_grad tensor yields a NON-LEAF (the ToCopy output), whose
   .grad never populates during backward — the optimizer then reads a zero
   gradient and silently no-ops, freezing F32 training at the init loss.
   The F64 param creators set requires_grad on the un-cast leaf (no cast), so
   they were unaffected; only the F32 path went cast-after-grad. */
static TensorHandle make_param_leaf(double* data, c10::IntArrayRef dims, torch::ScalarType dt) {
    auto t = torch::from_blob(data, dims, torch::kFloat64).clone();
    if (dt != torch::kFloat64) t = t.to(dt);
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

/* F64 — aliases to existing unsuffixed implementations.
   tensor_create_scalar_f64 and tensor_create_f64 are already defined
   above via the _impl refactor, so they're omitted here. */
TensorHandle tensor_create_1d_f64(int n, double* d, int rg)                             { return tensor_create_1d(n, d, rg); }
TensorHandle tensor_create_2d_f64(int rows, int cols, double* d, int rg)                { return tensor_create_2d(rows, cols, d, rg); }
TensorHandle tensor_create_param_1d_f64(int n, double* d)                               { return tensor_create_param_1d(n, d); }
TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* d)                  { return tensor_create_param_2d(rows, cols, d); }
TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* d)              { return tensor_create_param_3d(d0, d1, d2, d); }
TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* d)      { return tensor_create_param_4d(d0, d1, d2, d3, d); }
TensorHandle tensor_create_state_1d_f64(int n, double* d)                               { return tensor_create_state_1d(n, d); }
TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* d)                  { return tensor_create_state_2d(rows, cols, d); }

/* F32 — build at fp64 then cast down. Note: tensor_create_scalar_f32 and
   tensor_create_f32 already exist (refactored in their original location
   with _impl helpers); these wrappers cover the remaining 8 cases. */
TensorHandle tensor_create_1d_f32(int n, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    free(d);
    t = t.to(torch::kFloat32);
    if (rg) t.requires_grad_(true);   // cast-before-grad: keep the F32 tensor a leaf
    return from_tensor(std::move(t));
}
TensorHandle tensor_create_2d_f32(int rows, int cols, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(d);
    t = t.to(torch::kFloat32);
    if (rg) t.requires_grad_(true);   // cast-before-grad: keep the F32 tensor a leaf
    return from_tensor(std::move(t));
}
TensorHandle tensor_create_param_1d_f32(int n, double* d) {
    return make_param_leaf(d, {(int64_t)n}, torch::kFloat32);
}
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* d) {
    return make_param_leaf(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat32);
}
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* d) {
    return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat32);
}
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* d) {
    return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat32);
}
TensorHandle tensor_create_state_1d_f32(int n, double* d) {
    auto h = tensor_create_state_1d(n, d);
    return torch_cast_to(h, torch::kFloat32);
}
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* d) {
    auto h = tensor_create_state_2d(rows, cols, d);
    return torch_cast_to(h, torch::kFloat32);
}

TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    /* Returns a 0-dim view that shares storage with the parent tensor.
       Must be persistent — views into param tensors survive free_intermediates. */
    return from_tensor_persistent(to_tensor(h)->select(0, row).select(0, col));
}

TensorHandle tensor_view_1d(TensorHandle h, int idx) {
    return from_tensor_persistent(to_tensor(h)->select(0, idx));
}

/* tensor_item_1d / tensor_item_2d extracted to
   backend_torch/core/lifecycle/. */

/* The native optimizer surface (OptWrapper + create/free/step/zero_grad +
   set_lr / set_param_lr + clip_grad_value/norm + polyak_blend +
   optimizer_buf_count + get/set m/v/meta + native_train_step +
   optimizer_step_with_clip) lives in backend_torch/training/optimizer.cpp. */

/* tensor_lstm_gates lives in backend_torch/training/ntm_specific.cpp. */

/* tensor_lstm_gates_pair lives in
   backend_torch/nn/recurrent/lstm_gates_pair.cpp. */

/* ---------- System ---------- */

/* get_rss_mb / get_current_rss_mb live in shared_utils.c (compiled
 * once, unified symbol). Both are in the rename header's EXCLUDE
 * set so internal callers below resolve to the shared definitions. */

/* ---------- Backend Info ---------- */

const char* backend_name(void) { return "torch"; }

/* backend_memory_report removed (no Idris-side callers). */

void backend_reset_for_eval(void) {
    free_intermediates();
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (at::Tensor*)param_tensor(i_);
        if (tensor->grad().defined())
            tensor->grad().zero_();
    }
}

/* backend_epoch_begin / backend_profile_reset / backend_profile_report
   live in backend_torch/training/profiling.cpp. */

/* param_grad_item_at lives in shared/training/param_registry.c. */

/* ---------- Debug ---------- */

void tensor_print(TensorHandle h) {
    // std::cout << at::Tensor requires the tensor to live on CPU.
    std::cout << to_tensor(h)->cpu() << std::endl;
}

/* Job 3 Phase B — mx::compile is mlx-only; torch backend always reports
   disabled regardless of MLX_COMPILE env var. */
int  tensor_mlx_compile_enabled(void) { return 0; }
int  tensor_mlx_compile_invocations(void) { return 0; }
void tensor_mlx_compile_reset_stats(void) { }

/* ---------- Portable FFI helpers ---------- */

/* The 10 *_return helpers (tensor_backward_return, param_register_return,
   param_zero_all_grads_return, tensor_to_doubles_return,
   tensor_backward_conditional, tensor_backward_return_loss, idrisml_seq,
   backend_reset_for_eval_return, backend_profile_reset_return,
   backend_profile_report_return) live in shared/training/ffi_shims.c,
   compiled once with rename_torch.h to produce torch-suffixed symbols.
   Each shim only calls into FFI-exported functions declared in backend.h
   (every backend supplies its own implementation under those names), so
   no port methods are needed.

   native_train_step + optimizer_step_with_clip stay torch-local: they
   reach into OptWrapper internals (the prefix-scoped clip variants) that
   the shared trampolines don't carry. They'll lift when torch opts into
   the shared optimizer surface. */

/* tensor_live_count / tensor_peak_live_count live in
   backend_torch/training/intermediates.cpp.
   dropout_random_seed lives in shared_utils.c. */



/* ---- Inference-only dtype scaffolding (BF16, F16, Int, Bool) ----
   Generic dtype-parameterised create/cast over the lean non-grad set
   (scalar/create/1d/2d/cast). requires_grad is honored only for floating
   dtypes; torch rejects autograd on integer/bool. Grad param/state
   variants for these dtypes are deferred to the mixed-precision-training
   row; the f32/f64 param/state wrappers above are unchanged. */
static inline bool idrisml_is_floating_st(torch::ScalarType dt) {
    return dt == torch::kFloat32 || dt == torch::kFloat64 ||
           dt == torch::kBFloat16 || dt == torch::kHalf;
}
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


/* ---- Unified dtag-dispatch create/cast entry points ----
   One symbol per shape, dtag-keyed, superseding the per-dtype
   *_f32_streamed / *_f64_streamed / inference-dtype wrappers above. The
   Idris-side Scheme wrappers pass the RuntimeDType tag as the trailing
   `dtag` arg; the body switches internally. F32/F64 route to the existing
   dedicated creators (byte-identical behavior); other dtags route through
   the generic create_*_dt / make_param_leaf path. */
static torch::ScalarType st_for_dtag(int dtag) {
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

/* Per-shape dtag dispatchers bound into the port struct below. Renamed
   from `tensor_create_*_streamed` (FFI-named) — the shared
   shared/training/dtype_streamed.c TU now provides the FFI surface
   via port trampolines that strip the (mlx-only) stream_tag and call
   into these. F32/F64 route to the dedicated dtype creators
   (byte-identical with the previous in-file path); other dtags route
   through the generic create_*_dt / make_param_leaf / torch_cast_to
   path. */
static TensorHandle torch_create_scalar_dtag(double v, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_scalar_f32(v, rg);
        case 15: return tensor_create_scalar_f64(v, rg);
        default: return create_scalar_dt(v, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_dtag(double* data, int* shape, int rank, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_f32(data, shape, rank, rg);
        case 15: return tensor_create_f64(data, shape, rank, rg);
        default: return create_nd_dt(data, shape, rank, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_1d_dtag(int n, double* data, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_1d_f32(n, data, rg);
        case 15: return tensor_create_1d_f64(n, data, rg);
        default: return create_1d_dt(n, data, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_2d_dtag(int rows, int cols, double* data, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_2d_f32(rows, cols, data, rg);
        case 15: return tensor_create_2d_f64(rows, cols, data, rg);
        default: return create_2d_dt(rows, cols, data, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_1d_dtag(int n, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_1d_f32(n, data);
        case 15: return tensor_create_param_1d_f64(n, data);
        default: return make_param_leaf(data, {(int64_t)n}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_2d_dtag(int rows, int cols, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_2d_f32(rows, cols, data);
        case 15: return tensor_create_param_2d_f64(rows, cols, data);
        default: return make_param_leaf(data, {(int64_t)rows, (int64_t)cols}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_3d_f32(d0, d1, d2, data);
        case 15: return tensor_create_param_3d_f64(d0, d1, d2, data);
        default: return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_4d_f32(d0, d1, d2, d3, data);
        case 15: return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
        default: return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_state_1d_dtag(int n, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_1d_f32(n, data);
        case 15: return tensor_create_state_1d_f64(n, data);
        default: return torch_cast_to(tensor_create_state_1d(n, data), st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_state_2d_dtag(int rows, int cols, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_2d_f32(rows, cols, data);
        case 15: return tensor_create_state_2d_f64(rows, cols, data);
        default: return torch_cast_to(tensor_create_state_2d(rows, cols, data), st_for_dtag(dtag));
    }
}
static TensorHandle torch_cast_dtype_dtag(TensorHandle src, int dtag) {
    return from_tensor(to_tensor(src)->to(st_for_dtag(dtag)));
}

/* ---------- Shared training port adapter ----------
 *
 * Provides the per-tensor accessors that shared/training/param_registry.c
 * uses to talk to libtorch (numel / has_grad / grad read+write / zero /
 * data read+write / bulk doubles+int64 loaders). Other port slots stay
 * NULL until torch opts into the matching shared TUs (optimizer_*,
 * dtag-streamed creators, ffi_shims) — the shared trampolines never call
 * them because torch is excluded from those SHARED_BACKENDS_<tu> lists.
 *
 * Hot-path note: for F64/F32 contiguous CPU tensors, the element
 * accessors hit `data_ptr<>()` directly (one load); other dtype / device
 * combos route through the slow `.flatten().index({i}).cpu().item<>()`
 * path. Param storage is always contiguous + same device throughout a
 * run, so the fast path covers ~all live use. */

#include "shared/training/port.h"

/* Port-typed (void*) trampolines for the dtag creators. The internal
   torch_create_*_dtag helpers above return TensorHandle (which is
   void* at the C level — but C++ enforces the cast at the function-
   pointer-init site). */
static void* torch_port_create_scalar(double v, int rg, int dtag)                                 { return torch_create_scalar_dtag(v, rg, dtag); }
static void* torch_port_create(double* d, int* s, int r, int rg, int dtag)                        { return torch_create_dtag(d, s, r, rg, dtag); }
static void* torch_port_create_1d(int n, double* d, int rg, int dtag)                             { return torch_create_1d_dtag(n, d, rg, dtag); }
static void* torch_port_create_2d(int rows, int cols, double* d, int rg, int dtag)                { return torch_create_2d_dtag(rows, cols, d, rg, dtag); }
static void* torch_port_create_param_1d(int n, double* d, int dtag)                               { return torch_create_param_1d_dtag(n, d, dtag); }
static void* torch_port_create_param_2d(int rows, int cols, double* d, int dtag)                  { return torch_create_param_2d_dtag(rows, cols, d, dtag); }
static void* torch_port_create_param_3d(int d0, int d1, int d2, double* d, int dtag)              { return torch_create_param_3d_dtag(d0, d1, d2, d, dtag); }
static void* torch_port_create_param_4d(int d0, int d1, int d2, int d3, double* d, int dtag)      { return torch_create_param_4d_dtag(d0, d1, d2, d3, d, dtag); }
static void* torch_port_create_state_1d(int n, double* d, int dtag)                               { return torch_create_state_1d_dtag(n, d, dtag); }
static void* torch_port_create_state_2d(int rows, int cols, double* d, int dtag)                  { return torch_create_state_2d_dtag(rows, cols, d, dtag); }
static void* torch_port_cast_dtype(void* src, int dtag)                                            { return torch_cast_dtype_dtag((TensorHandle)src, dtag); }

static int torch_port_tensor_numel(void* h) {
    return (int)to_tensor(h)->numel();
}

static int torch_port_tensor_requires_grad(void* h) {
    return to_tensor(h)->requires_grad() ? 1 : 0;
}

static int torch_port_tensor_has_grad(void* h) {
    return to_tensor(h)->mutable_grad().defined() ? 1 : 0;
}

static double torch_port_grad_read(void* h, int i) {
    auto* t = to_tensor(h);
    auto& g = t->mutable_grad();
    if (!g.defined()) return 0.0;
    if (g.is_cpu() && g.is_contiguous()) {
        if (g.dtype() == torch::kFloat64) return ((double*)g.data_ptr())[i];
        if (g.dtype() == torch::kFloat32) return (double)((float*)g.data_ptr())[i];
    }
    return g.flatten().index({i}).cpu().item<double>();
}

static void torch_port_grad_write(void* h, int i, double v) {
    auto* t = to_tensor(h);
    auto& g = t->mutable_grad();
    if (!g.defined()) return;
    if (g.is_cpu() && g.is_contiguous()) {
        if (g.dtype() == torch::kFloat64) { ((double*)g.data_ptr())[i] = v; return; }
        if (g.dtype() == torch::kFloat32) { ((float*)g.data_ptr())[i] = (float)v; return; }
    }
    g.flatten().index_put_({i}, v);
}

static void torch_port_zero_grad(void* h) {
    auto& g = to_tensor(h)->mutable_grad();
    if (g.defined()) g.zero_();
}

static double torch_port_data_read(void* h, int i) {
    auto* t = to_tensor(h);
    if (t->is_cpu() && t->is_contiguous()) {
        if (t->dtype() == torch::kFloat64) return ((double*)t->data_ptr())[i];
        if (t->dtype() == torch::kFloat32) return (double)((float*)t->data_ptr())[i];
    }
    return t->flatten().index({i}).cpu().item<double>();
}

static void torch_port_data_write(void* h, int i, double v) {
    auto* t = to_tensor(h);
    if (t->is_cpu() && t->is_contiguous()) {
        if (t->dtype() == torch::kFloat64) { ((double*)t->data_ptr())[i] = v; return; }
        if (t->dtype() == torch::kFloat32) { ((float*)t->data_ptr())[i] = (float)v; return; }
    }
    torch::NoGradGuard no_grad;
    t->flatten().index_put_({i}, v);
}

static void torch_port_load_doubles(void* h, const double* src, int n) {
    torch::NoGradGuard no_grad;
    auto* t = to_tensor(h);
    auto staging = torch::from_blob(const_cast<double*>(src), {(int64_t)n}, torch::kFloat64);
    t->view({n}).copy_(staging);
}

static void torch_port_load_int64(void* h, const int64_t* src, int n) {
    torch::NoGradGuard no_grad;
    auto* t = to_tensor(h);
    auto staging = torch::from_blob(const_cast<int64_t*>(src), {(int64_t)n}, torch::kInt64);
    t->view({n}).copy_(staging);
}

const BackendPort g_active_port = {
    /* Tensor introspection + per-element access + bulk grad/data ops:
       supplied by the torch port shims above. */
    .tensor_numel              = torch_port_tensor_numel,
    .tensor_requires_grad      = torch_port_tensor_requires_grad,
    .tensor_has_grad           = torch_port_tensor_has_grad,
    .data_read                 = torch_port_data_read,
    .data_write                = torch_port_data_write,
    .grad_read                 = torch_port_grad_read,
    .grad_write                = torch_port_grad_write,
    .zero_grad                 = torch_port_zero_grad,
    .load_doubles              = torch_port_load_doubles,
    .load_int64                = torch_port_load_int64,
    /* Slots whose shared TUs torch hasn't opted into yet — see the
       SHARED_BACKENDS_<tu> lists in the Makefile. These stay nullptr
       until torch's adapter ships their bindings AND the shared TU
       gets compiled for torch. Ordering matches port.h's struct
       declaration order (C++ ISO-required for designated init). */
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
    /* Dtag-streamed creators: torch supplies its libtorch-backed
       dtag dispatchers via the torch_port_create_* shims above. */
    .create_scalar             = torch_port_create_scalar,
    .create                    = torch_port_create,
    .create_1d                 = torch_port_create_1d,
    .create_2d                 = torch_port_create_2d,
    .create_param_1d           = torch_port_create_param_1d,
    .create_param_2d           = torch_port_create_param_2d,
    .create_param_3d           = torch_port_create_param_3d,
    .create_param_4d           = torch_port_create_param_4d,
    .create_state_1d           = torch_port_create_state_1d,
    .create_state_2d           = torch_port_create_state_2d,
    .cast_dtype                = torch_port_cast_dtype,
};

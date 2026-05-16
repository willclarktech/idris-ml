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

/* MPS eager-init constructor lives in backend_torch/mps_init.cpp. */


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
/* make_param_leaf + the dtag-dispatch surface are declared in
   backend_torch/training/dtype_dispatch.h. Included early because
   tensor_create_param_3d below (and tensor_one_hot) call into it. */
#include "backend_torch/training/dtype_dispatch.h"

/* ---------- Lifecycle ----------
   tensor_create_scalar* / tensor_create* / tensor_cast_dtype_* extracted
   to backend_torch/core/lifecycle/{create_scalar,create,cast}.cpp. */

/* tensor_clone / tensor_free / tensor_retain_handle / tensor_release_handle
   extracted to backend_torch/core/lifecycle/. The `freed_by_cleanup`
   set + free_intermediates() impl live in
   backend_torch/training/intermediates.cpp. */

/* Accessors (numel / dim / size / to_doubles / to_floats / to_int64 /
 * dtype_name) live in backend_torch/core/lifecycle/accessors.cpp.
 * tensor_item / tensor_item_1d / tensor_item_2d live in their own
 * core/lifecycle/ TUs. */

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

/* tensor_create_param_3d lives in backend_torch/core/lifecycle/create_param_state.cpp. */

/* Shape ops live in backend_torch/linear/shape/.
 * Stack / cat live in backend_torch/linear/concat/. */
/* Autograd surface (tensor_backward + tensor_grad/zero_grad/detach/with_grad/
 * set_requires_grad/no_grad_begin/end/epoch_begin/end) lives in
 * backend_torch/training/autograd.cpp. */

/* tensor_to_device + tensor_device live in backend_torch/device.cpp. */


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

/* tensor_to_int64 lives in backend_torch/core/lifecycle/accessors.cpp. */

/* tensor_subtract_scalar_inplace lives in
   backend_torch/training/ntm_specific.cpp. */

/* tensor_create_1d + tensor_create_2d (F64 base) live in
 * backend_torch/core/lifecycle/create_param_state.cpp.
 *
 * tensor_alloc_doubles / tensor_free_doubles / tensor_read_double /
 * tensor_ptr_array_alloc live in shared_utils.c.
 *
 * tensor_stack_from_array / tensor_cat_from_array live in
 * backend_torch/linear/concat/{stack,cat}.cpp. */


/* st_for_dtag + the unified create/cast dtag dispatchers + the F32/F64
   explicit-suffix wrappers + make_param_leaf + idrisml_is_floating_st all
   live in backend_torch/training/dtype_dispatch.cpp. tensor_one_hot below
   calls st_for_dtag via the dtype_dispatch.h declaration (included
   above). */

/* tensor_one_hot + tensor_batch + tensor_unbatch live in
 * backend_torch/core/lifecycle/batch.cpp.
 * tensor_create_{1d,2d}, tensor_create_param_{1,2,3,4}d, tensor_create_state_{1,2}d,
 * tensor_view_{1d,2d} live in backend_torch/core/lifecycle/create_param_state.cpp.
 * Per-dtype F32 variants + dtag dispatchers live in
 * backend_torch/training/dtype_dispatch.cpp. */

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
/* The dtype-dispatch block (Inference-only dtype scaffolding, st_for_dtag,
 * create_*_dt helpers, and the torch_create_*_dtag / torch_cast_dtype_dtag
 * family) lives in backend_torch/training/dtype_dispatch.cpp — declared in
 * its header which the monolith includes above. */


/* Shared training port adapter (torch_port_* shims + g_active_port struct)
 * lives in backend_torch/training/adapter.cpp. */


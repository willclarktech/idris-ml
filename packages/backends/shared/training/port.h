/* shared/training/port.h — backend port surface for shared training code.
 *
 * Defines the function-pointer dispatch table that shared training-side
 * code (optimizer, param_registry, dtag_dispatch, ffi_shims, profiler)
 * calls into. Each backend supplies one `const BackendPort g_active_port`
 * instance via its `<backend>/training/adapter.<c|cpp>`. Shared TUs
 * dereference `g_active_port` directly.
 *
 * Why a struct of function pointers rather than weak symbols + dlsym?
 * Two reasons:
 *  1. Multi-link dylibs (`BACKEND=tape,torch,mlx`) compile shared/training/
 *     TUs once per backend (each with its own rename header), so each
 *     backend gets its own `g_active_port_<b>` symbol with no name
 *     collision. Function-pointer dispatch handles this naturally —
 *     weak-symbol fallback would require ifdefs per backend.
 *  2. The dispatch table is a single allocator-free struct literal;
 *     debugging adapter mis-wiring is trivial (compare struct contents).
 *
 * Surface kept minimal: only the tensor accesses the lifted code actually
 * makes. F64 byte-identical guarantee preserved by routing per-element
 * reads/writes through `data_read`/`data_write` (dtype-aware: F32
 * narrowing matches `tape_load_d`/`tape_store_d` precisely). */

#ifndef SHARED_TRAINING_PORT_H
#define SHARED_TRAINING_PORT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BackendPort {
  /* ----------------------------------------------------------------------
     Tensor introspection. `t` is the backend's TensorHandle (opaque void*
     here — backends downcast to their concrete Tensor type internally).
     ---------------------------------------------------------------------- */
  int    (*tensor_numel)(void* t);
  int    (*tensor_requires_grad)(void* t);
  int    (*tensor_has_grad)(void* t);

  /* ----------------------------------------------------------------------
     Per-element dtype-aware read/write. F32 storage narrows/widens
     through these (tape's `tape_load_d`/`tape_store_d` semantics);
     F64 hits the raw buffer. Used by the optimizer's per-element
     update loop and by polyak_blend.
     ---------------------------------------------------------------------- */
  double (*data_read)(void* t, int i);
  void   (*data_write)(void* t, int i, double v);
  double (*grad_read)(void* t, int i);
  void   (*grad_write)(void* t, int i, double v);

  /* ----------------------------------------------------------------------
     Bulk grad zero (one whole tensor's grad buffer set to 0).
     ---------------------------------------------------------------------- */
  void   (*zero_grad)(void* t);

  /* ----------------------------------------------------------------------
     Bulk data load (for safetensors load + param loaders).
     load_doubles writes `src[i]` to data[i] dtype-aware;
     load_int64 widens to double via the dtype's lingua-franca path.
     ---------------------------------------------------------------------- */
  void   (*load_doubles)(void* t, const double* src, int n);
  void   (*load_int64)(void* t, const int64_t* src, int n);

  /* ----------------------------------------------------------------------
     Backward driver. Triggers gradient propagation from `loss` to all
     `requires_grad=1` ancestors. Tape: walks the Wengert list in
     reverse via the op_dispatch table. Torch/mlx: delegate to their
     native autograd.
     ---------------------------------------------------------------------- */
  void   (*backward)(void* loss);


  /* ----------------------------------------------------------------------
     Optimizer surface. Each backend defines its own optimizer struct
     and supplies all of these (tape: flat-buffer Optimizer struct +
     per-element SGD/RMSprop/Adam/AdamW math; torch: libtorch
     OptWrapper wrapping torch::optim::Adam + at::_foreach_adam math;
     mlx: TBD). The shared/training/optimizer.c file provides tiny
     trampolines from the FFI-named entry points (`optimizer_create_*`,
     etc.) to these port methods.

     The shared file still owns the cross-cutting helpers that don't
     touch optimizer state: `optimizer_zero_grad` (delegates to
     param_zero_all_grads), `polyak_blend` and `optimizer_clip_*`
     (per-element via the port's grad/data accessors),
     `native_train_step` / `optimizer_step_with_clip` (high-level
     wrappers that compose zero_grad/backward/clip/step).
     ---------------------------------------------------------------------- */

  /* Constructors — each returns a backend-owned struct cast to void*. */
  void* (*optimizer_create_sgd)(double lr);
  void* (*optimizer_create_rmsprop)(double lr, double alpha, double eps,
                                     double weight_decay, double momentum);
  void* (*optimizer_create_adam)(double lr, double beta1, double beta2, double eps);
  void* (*optimizer_create_adam_group)(double lr, double beta1, double beta2,
                                        double eps, const char* prefix);
  void* (*optimizer_create_adamw)(double lr, double beta1, double beta2, double eps,
                                   double weight_decay);

  /* Lifecycle / setters. */
  void  (*optimizer_free)(void* opt);
  void  (*optimizer_set_lr)(void* opt, double lr);
  void  (*optimizer_set_param_lr)(void* opt, const char* name, double lr);

  /* Per-step math. Adapter is responsible for ALL backend hygiene
     (intermediate cleanup, prof_* updates, tape_reset where applicable). */
  void  (*optimizer_step)(void* opt);

  /* Prefix-scoped grad clipping. Each backend's Optimizer holds the
     prefix that scopes which params it owns (SAC's multi-optimizer
     setup: actor_/q1_/q2_/etc.); these methods clip only the owned
     params. Distinct from the global `optimizer_clip_grad_value` /
     `optimizer_clip_grad_norm` exported as FFI from
     shared/training/optimizer.c. `native_train_step` and
     `optimizer_step_with_clip` route through these so multi-optimizer
     training (SAC etc.) preserves the scoping. */
  void   (*optimizer_clip_grad_value_filtered)(void* opt, double max_val);
  double (*optimizer_clip_grad_norm_filtered)(void* opt, double max_norm);

  /* Serialization — flat-buffer m/v access + 9-double meta vector
     (type, lr, β1, β2, eps, alpha, weight_decay, momentum, t).
     Out-buffers are caller-allocated. */
  int   (*optimizer_buf_count)(void* opt);
  void  (*optimizer_get_m)(void* opt, int idx, double* out);
  void  (*optimizer_get_v)(void* opt, int idx, double* out);
  void  (*optimizer_set_m)(void* opt, int idx, const double* in);
  void  (*optimizer_set_v)(void* opt, int idx, const double* in);
  void  (*optimizer_get_meta)(void* opt, double* out9);
  void  (*optimizer_set_meta)(void* opt, const double* in9);

  /* ----------------------------------------------------------------------
     Wall clock for profiler. Returns milliseconds since some epoch
     (only deltas matter). Tape, torch, and mlx all use the same
     gettimeofday-based implementation today.
     ---------------------------------------------------------------------- */
  double (*wall_ms)(void);

  /* ----------------------------------------------------------------------
     Dtag-dispatched create / cast surface. Each method takes the
     same arguments as the corresponding `tensor_create_*_streamed`
     FFI wrapper minus the `stream_tag` (an mlx-only knob the shared
     wrappers absorb at the boundary). The adapter picks the right
     backend storage variant (F64 lingua franca, real F32 storage,
     or lingua-franca-rounded for the other dtags) based on `dtag`.
     `data` buffer ownership matches the underlying creator (tape
     copies + frees; torch/mlx wrap their own storage).
     ---------------------------------------------------------------------- */
  void* (*create_scalar)(double v, int rg, int dtag);
  void* (*create)(double* data, int* shape, int rank, int rg, int dtag);
  void* (*create_1d)(int n, double* data, int rg, int dtag);
  void* (*create_2d)(int rows, int cols, double* data, int rg, int dtag);
  void* (*create_param_1d)(int n, double* data, int dtag);
  void* (*create_param_2d)(int rows, int cols, double* data, int dtag);
  void* (*create_param_3d)(int d0, int d1, int d2, double* data, int dtag);
  void* (*create_param_4d)(int d0, int d1, int d2, int d3, double* data, int dtag);
  void* (*create_state_1d)(int n, double* data, int dtag);
  void* (*create_state_2d)(int rows, int cols, double* data, int dtag);
  void* (*cast_dtype)(void* src, int dtag);
} BackendPort;

/* Each backend defines exactly one instance with internal linkage at
   adapter.c file scope; the external symbol is renamed per-backend by
   the build's rename header when multi-link demands it. For single-
   backend builds the symbol is `g_active_port` outright. */
extern const BackendPort g_active_port;

#ifdef __cplusplus
}
#endif

#endif /* SHARED_TRAINING_PORT_H */

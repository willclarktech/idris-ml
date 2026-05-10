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
     Epoch boundary — called by the DEFAULT shared optimizer at the
     end of `optimizer_step()` (i.e. when `optimizer_step` is NULL
     and the per-element flat-buffer loop ran). `step_start_ms` is
     the wall-clock at start of step() so the adapter can credit
     (per-param loop + post-step hygiene) to its `prof_optimizer_ms`.

     Tape: dumps DEBUG_LSTM_TRAJ if enabled → tape_reset() →
       re-register every param on the fresh tape (so their grads
       stay reachable after the reset) → bump prof epoch counters.
     Adapters that override `optimizer_step` don't see this callback —
     they're responsible for their own epoch hygiene.
     ---------------------------------------------------------------------- */
  void   (*epoch_boundary)(double step_start_ms);

  /* ----------------------------------------------------------------------
     Optional optimizer-step override. NULL = use the shared default
     per-element flat-buffer loop (tape's case). Set to a backend-
     supplied function when the backend's native math doesn't match
     the shared loop — e.g. libtorch's `at::_foreach_adam` fused
     multi-tensor primitives in torch, or mlx's vectorized in-graph
     update. The override sees the full Optimizer struct (defined in
     shared/training/optimizer.h) and is responsible for ALL backend
     hygiene (intermediate cleanup, prof_* updates, etc.).
     ---------------------------------------------------------------------- */
  void   (*optimizer_step)(void* opt_handle);

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

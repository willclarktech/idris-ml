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
     Epoch boundary. Tape: tape_reset() + re-register params on the
     fresh tape (so their grads stay reachable). Torch/mlx: no-op —
     their autograd doesn't carry a persistent Wengert list between
     epochs.
     ---------------------------------------------------------------------- */
  void   (*epoch_boundary)(void);

  /* ----------------------------------------------------------------------
     Wall clock for profiler. Returns milliseconds since some epoch
     (only deltas matter). Tape, torch, and mlx all use the same
     gettimeofday-based implementation today.
     ---------------------------------------------------------------------- */
  double (*wall_ms)(void);
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

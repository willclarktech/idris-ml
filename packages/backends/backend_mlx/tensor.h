/* Tensor representation for the mlx backend's modular tree.
 *
 * Per-op .cpp files under backend_mlx/ include this header to access
 * the Tensor struct (the mlx-side autograd handle wrapping an mx::array
 * plus refcount + tape index + grad), the tensor-tracking globals it
 * mutates, and the retain/release helpers used to drive lifetime.
 *
 * Surfaces NOT exposed here:
 *   - tape mechanics → backend_mlx/tape.h
 *   - stream selection / WITH_STREAM macro → backend_mlx/stream.h
 *   - F32↔F64 / mx::array ↔ host-double bridge → backend_mlx/precision.h
 */
#ifndef IDRISML_BACKEND_MLX_TENSOR_H
#define IDRISML_BACKEND_MLX_TENSOR_H

#include <mlx/mlx.h>
#include <vector>
#include "../backend.h"

namespace mx = mlx::core;

/* TensorPair is a typedef-struct from backend.h — complete type
   already in scope via the include above. */

struct Tensor {
    mx::array data;
    mx::array grad;
    bool requires_grad;
    bool has_grad;
    int tape_idx;
    int pool_idx;    /* unique index for replay pool */
    /* Reference count = number of long-term holders pointing at this
       Tensor: the Idris-side wrap, each tape entry that has it as an
       arg, the param_registry, etc. Ctor sets it to 0 — the first
       holder takes the first retain. When the count drops to 0, the
       Tensor is removed from all_tensors and deleted, freeing the
       underlying mx::array → MetalAllocator releases the MTLBuffer. */
    int refcount;
    long create_id;  /* generation marker: g_mlx_create_calls_global at construction */

    Tensor(mx::array d, bool rg = false);
};

/* Tracking globals — defined in backend_mlx.cpp, referenced from any
   TU that constructs a Tensor (the ctor body inlines and touches all
   four). */
extern std::vector<Tensor*> all_tensors;
extern std::vector<TensorPair*> all_pairs;
extern int next_pool_idx;
extern long g_mlx_create_calls_global;
extern long g_mlx_peak_live;

/* Refcount machinery — unconditional. Every Tensor's lifecycle is
   driven by refcount: the Idris-side wrap retains on creation,
   tape_append retains as args, param_register retains. Symmetric
   releases (wrap drain, tape_reset, param_clear). When the count drops
   to 0, the Tensor's slot in all_tensors is reclaimed by the sweep at
   the next tape_reset / no_grad_end (we don't delete on release — that
   would invalidate any in-flight `Tensor*` arg currently mid-call). */
void tensor_retain_internal(Tensor* t);
void tensor_release_internal(Tensor* t);

#endif /* IDRISML_BACKEND_MLX_TENSOR_H */

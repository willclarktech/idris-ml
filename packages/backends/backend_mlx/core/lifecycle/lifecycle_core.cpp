/* Tensor lifecycle core — mlx.
 *
 * Holds the canonical definitions of:
 *
 *   - the four tracking globals (all_tensors, all_pairs, next_pool_idx,
 *     g_mlx_create_calls_global, g_mlx_peak_live), all declared extern
 *     in tensor.h so per-op TUs in the modular tree can reach them.
 *   - the Tensor constructor — assigns pool_idx, push_back to
 *     all_tensors, bumps the peak-live counter.
 *   - tensor_retain_internal / tensor_release_internal (refcount
 *     primitives used by tape_append + the FFI lifecycle).
 *   - tensor_retain_handle / tensor_release_handle (C-exported variants
 *     for the Idris-side guardian + Scheme drain).
 */
#include "../../tensor.h"

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

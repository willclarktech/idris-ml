/* Dtype dispatch for the torch backend's modular tree.
 *
 * Exposes:
 *   - `idrisml_is_floating_st` — predicate used by tensor_set_requires_grad
 *     in the monolith (libtorch throws if you set requires_grad on a
 *     non-floating tensor).
 *   - `make_param_leaf` — the cast-before-requires_grad helper used by
 *     tensor_create_param_{1,2,3,4}d (F64 path) and the F32 param creators.
 *   - `torch_cast_to` — F32 state creator helper (cast an existing
 *     F64 state tensor down to F32).
 *   - `st_for_dtag` — runtime dtag → torch::ScalarType lookup; used by
 *     tensor_one_hot and the dtag dispatchers below.
 *   - `torch_create_*_dtag` / `torch_cast_dtype_dtag` — the per-shape
 *     dtag dispatchers wired into the shared training port (adapter.cpp).
 *
 * F32/F64 explicit-suffix variants (`tensor_create_*_f32` /
 * `tensor_create_*_f64`) live in dtype_dispatch.cpp as well; they're not
 * declared here because their FFI names are reachable via backend.h /
 * rename headers — only intra-tree callers need declarations.
 */
#ifndef IDRISML_BACKEND_TORCH_DTYPE_DISPATCH_H
#define IDRISML_BACKEND_TORCH_DTYPE_DISPATCH_H

#include <torch/torch.h>
#include "../tensor.h"

bool idrisml_is_floating_st(torch::ScalarType dt);

torch::ScalarType st_for_dtag(int dtag);

TensorHandle make_param_leaf(double* data, c10::IntArrayRef dims, torch::ScalarType dt);
TensorHandle torch_cast_to(TensorHandle h, torch::ScalarType dt);

/* Dtag-keyed dispatchers — wired into the shared port via adapter.cpp. */
TensorHandle torch_create_scalar_dtag(double v, int rg, int dtag);
TensorHandle torch_create_dtag(double* data, int* shape, int rank, int rg, int dtag);
TensorHandle torch_create_1d_dtag(int n, double* data, int rg, int dtag);
TensorHandle torch_create_2d_dtag(int rows, int cols, double* data, int rg, int dtag);
TensorHandle torch_create_param_1d_dtag(int n, double* data, int dtag);
TensorHandle torch_create_param_2d_dtag(int rows, int cols, double* data, int dtag);
TensorHandle torch_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag);
TensorHandle torch_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data, int dtag);
TensorHandle torch_create_state_1d_dtag(int n, double* data, int dtag);
TensorHandle torch_create_state_2d_dtag(int rows, int cols, double* data, int dtag);
TensorHandle torch_cast_dtype_dtag(TensorHandle src, int dtag);

#endif /* IDRISML_BACKEND_TORCH_DTYPE_DISPATCH_H */

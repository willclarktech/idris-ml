/* tensor_create variants for the torch backend.
 *
 * Per-dtype creators (`_f32` / `_f64`) plus the legacy unsuffixed entry
 * (routes to f64). The static impl uses libtorch's from_blob (zero-copy
 * view over the caller's buffer) and .clone() so the tensor owns its
 * storage. .to(dt) casts to the target dtype when not f64. Result is
 * persistent — survives optimizer_step. */
#include "../../tensor.h"

#include <vector>

static TensorHandle tensor_create_impl(double* data, int* shape, int rank,
                                       int requires_grad, torch::ScalarType dt) {
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; i++) dims[i] = shape[i];
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto t = torch::from_blob(data, dims, opts).clone();
    if (dt != torch::kFloat64) t = t.to(dt);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}

extern "C" TensorHandle tensor_create_f32(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_impl(data, shape, rank, requires_grad, torch::kFloat32);
}

extern "C" TensorHandle tensor_create_f64(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_impl(data, shape, rank, requires_grad, torch::kFloat64);
}

extern "C" TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    return tensor_create_f64(data, shape, rank, requires_grad);
}

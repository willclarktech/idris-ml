/* tensor_create_scalar variants for the torch backend.
 *
 * Per-dtype creators (`_f32` / `_f64`) plus the legacy unsuffixed entry
 * (routes to f64, current historical behaviour). The dtype is selected
 * by a torch::ScalarType built locally; the result is constructed as a
 * persistent tensor (not tracked in the intermediates vector — survives
 * optimizer_step). */
#include "../../tensor.h"

static TensorHandle tensor_create_scalar_impl(double value, int requires_grad, torch::ScalarType dt) {
    auto t = torch::tensor(value, torch::dtype(dt));
    if (requires_grad) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}

extern "C" TensorHandle tensor_create_scalar_f32(double value, int requires_grad) {
    return tensor_create_scalar_impl(value, requires_grad, torch::kFloat32);
}

extern "C" TensorHandle tensor_create_scalar_f64(double value, int requires_grad) {
    return tensor_create_scalar_impl(value, requires_grad, torch::kFloat64);
}

extern "C" TensorHandle tensor_create_scalar(double value, int requires_grad) {
    return tensor_create_scalar_f64(value, requires_grad);
}

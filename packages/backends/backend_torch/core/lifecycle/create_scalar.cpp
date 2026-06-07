/* tensor_create_scalar variants for the torch backend.
 *
 * Per-dtype creators (`_f32` / `_f64`) plus the legacy unsuffixed entry
 * (routes to f64, current historical behaviour). The dtype is selected
 * by a torch::ScalarType built locally; the result is constructed as a
 * persistent tensor (not tracked in the intermediates vector — survives
 * optimizer_step). Migrates to `g_torch_target_device` (set at dylib
 * load from TORCH_DEVICE) so torch-mps / torch-cuda builds land scalars
 * on the right hardware — without this the streamed-path scalars
 * silently stayed on CPU regardless of TORCH_DEVICE. requires_grad set
 * AFTER cast+move so the result is a leaf. */
#include "../../tensor.h"

extern c10::Device g_torch_target_device;

static TensorHandle tensor_create_scalar_impl(double value, int requires_grad,
                                              torch::ScalarType dt) {
	auto t = torch::tensor(value, torch::dtype(dt));
	// Effective target degrades to CPU on (MPS, F64) — Metal rejects F64.
	c10::Device target =
	    (g_torch_target_device.type() == c10::DeviceType::MPS && dt == torch::kFloat64)
	        ? at::kCPU
	        : g_torch_target_device;
	if (target != at::kCPU) t = t.to(target);
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

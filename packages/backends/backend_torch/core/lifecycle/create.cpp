/* tensor_create variants for the torch backend.
 *
 * Per-dtype creators (`_f32` / `_f64`) plus the legacy unsuffixed entry
 * (routes to f64). The static impl uses libtorch's from_blob (zero-copy
 * view over the caller's buffer) and .clone() so the tensor owns its
 * storage. Cast + device migration are combined into a single .to(opts)
 * call when either differs from F64-on-CPU — keeps the leaf transition
 * atomic (cast/move-before-requires_grad). Migrates to
 * `g_torch_target_device` (set at dylib load from TORCH_DEVICE) so
 * torch-mps / torch-cuda builds land tensors on the right hardware. */
#include "../../tensor.h"

#include <vector>

extern c10::Device g_torch_target_device;

static TensorHandle tensor_create_impl(double* data, int* shape, int rank, int requires_grad,
                                       torch::ScalarType dt) {
	std::vector<int64_t> dims(rank);
	for (int i = 0; i < rank; i++)
		dims[i] = shape[i];
	auto opts0 = torch::TensorOptions().dtype(torch::kFloat64);
	auto t = torch::from_blob(data, dims, opts0).clone();
	// Effective target degrades to CPU on (MPS, F64) — Metal rejects F64
	// at construction. Lets Transfer.idr explicitly create F64-on-CPU even
	// under a `TORCH_DEVICE=mps` build, then migrate later with the typed
	// `toExecutor` (which gates on `Compatible`).
	c10::Device target =
	    (g_torch_target_device.type() == c10::DeviceType::MPS && dt == torch::kFloat64)
	        ? at::kCPU
	        : g_torch_target_device;
	bool need_cast = dt != torch::kFloat64;
	bool need_move = target != at::kCPU;
	if (need_cast || need_move) {
		auto opts = torch::TensorOptions().dtype(dt).device(target);
		t = t.to(opts);
	}
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

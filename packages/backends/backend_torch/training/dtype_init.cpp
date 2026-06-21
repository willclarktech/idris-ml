/* Fused param create + init for the torch backend.
 *
 * Background: Idris's HF model state construction used to fill each
 * parameter tensor element-by-element on the host side via `traverse
 * normalSample` + `packDoubles` (per-element `prim__setDouble` FFI),
 * costing 58 minutes for Llama-3.2-1B's 1.24B parameters (~30 min in
 * Box-Muller in Chez + ~28 min in per-element FFI). PyTorch's
 * equivalent path does in-place init in C++ at memory-bandwidth
 * speed — ~1 ms per tensor for a 1B-param model.
 *
 * This file provides the C-side fused alternative: each
 * `torch_create_param_*_<init>_dtag` allocates a fresh torch::Tensor
 * (uninitialised at F64/CPU), applies an in-place init kernel from
 * `torch::nn::init` (normal) or `t.fill_` (const), then runs the
 * standard make-param-leaf migration (cast + device move + autograd
 * leaf discipline). All work happens inside libtorch — no host loop,
 * no per-element FFI.
 *
 * Wired into the shared training port (adapter.cpp's g_active_port);
 * reached from Idris via the `tensor_create_param_*_<init>_streamed`
 * FFI wrappers in shared/training/dtype_streamed.c.
 *
 * Leaf discipline mirrors `make_param_leaf` in dtype_dispatch.cpp:
 * cast + move MUST precede requires_grad_, otherwise the resulting
 * tensor is non-leaf and .grad never populates (the optimizer would
 * silently no-op, freezing training at the init loss). See
 * gotchas.md "A parameter must be cast/moved before requires_grad_".
 */

#include <torch/torch.h>
#include <cstdint>
#include "dtype_dispatch.h"
#include "../tensor.h"

namespace {

/* Empty + init + leaf-discipline helper. The init lambda runs in-place
   on an F64/CPU tensor; after init, the tensor is migrated/cast to the
   requested target dtype and device, then marked requires_grad. */
template <typename Init>
TensorHandle make_param_leaf_empty(c10::IntArrayRef dims, torch::ScalarType dt, Init init) {
	auto t = torch::empty(dims, torch::TensorOptions().dtype(torch::kFloat64).device(at::kCPU));
	init(t);
	c10::Device target = torch_effective_device(dt);
	const bool need_cast = dt != torch::kFloat64;
	const bool need_move = target != at::kCPU;
	if (need_cast || need_move) {
		auto opts = torch::TensorOptions().dtype(dt).device(target);
		t = t.to(opts);
	}
	t.requires_grad_(true);
	TORCH_CHECK(t.is_leaf(),
	            "parameter tensor is not an autograd leaf (fused-init path): cast/move "
	            "(.to(dtype/device)) must precede requires_grad_, otherwise .grad never "
	            "populates and the optimizer silently freezes training");
	return from_tensor_persistent(std::move(t));
}

} // namespace

/* ---- Normal(mean, std) initialisation ---- */
TensorHandle torch_create_param_1d_normal_dtag(int n, double mean, double std, int dtag) {
	return make_param_leaf_empty({(int64_t)n}, st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { torch::nn::init::normal_(t, mean, std); });
}

TensorHandle torch_create_param_2d_normal_dtag(int rows, int cols, double mean, double std,
                                               int dtag) {
	return make_param_leaf_empty({(int64_t)rows, (int64_t)cols}, st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { torch::nn::init::normal_(t, mean, std); });
}

TensorHandle torch_create_param_3d_normal_dtag(int d0, int d1, int d2, double mean, double std,
                                               int dtag) {
	return make_param_leaf_empty({(int64_t)d0, (int64_t)d1, (int64_t)d2}, st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { torch::nn::init::normal_(t, mean, std); });
}

TensorHandle torch_create_param_4d_normal_dtag(int d0, int d1, int d2, int d3, double mean,
                                               double std, int dtag) {
	return make_param_leaf_empty({(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3},
	                             st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { torch::nn::init::normal_(t, mean, std); });
}

/* ---- Constant fill ---- */
TensorHandle torch_create_param_1d_const_dtag(int n, double value, int dtag) {
	return make_param_leaf_empty({(int64_t)n}, st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { t.fill_(value); });
}

TensorHandle torch_create_param_2d_const_dtag(int rows, int cols, double value, int dtag) {
	return make_param_leaf_empty({(int64_t)rows, (int64_t)cols}, st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { t.fill_(value); });
}

TensorHandle torch_create_param_3d_const_dtag(int d0, int d1, int d2, double value, int dtag) {
	return make_param_leaf_empty({(int64_t)d0, (int64_t)d1, (int64_t)d2}, st_for_dtag(dtag),
	                             [=](torch::Tensor& t) { t.fill_(value); });
}

TensorHandle torch_create_param_4d_const_dtag(int d0, int d1, int d2, int d3, double value,
                                              int dtag) {
	return make_param_leaf_empty({(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3},
	                             st_for_dtag(dtag), [=](torch::Tensor& t) { t.fill_(value); });
}

/* ---- Init RNG seed ----
   `torch::manual_seed` seeds the CPU RNG used by normal_/uniform_/etc.
   above. Per-device RNGs (MPS, CUDA) are seeded separately by libtorch
   when needed, but since our init pipeline runs init on CPU then
   migrates, the CPU seed is what governs determinism here. */
void torch_set_init_seed(uint64_t seed) {
	torch::manual_seed((uint64_t)seed);
}

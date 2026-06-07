/* Backend meta surface — torch.
 *
 *   - backend_name              "torch"
 *   - backend_reset_for_eval    free_intermediates + zero every param's
 *                               grad. Fired between training phases (eval
 *                               loops, checkpoint reload).
 *   - tensor_print              std::cout the .cpu()-routed tensor.
 *   - tensor_mlx_compile_*      mx::compile is mlx-only; the torch
 *                               backend reports disabled / zero
 *                               invocations regardless of MLX_COMPILE.
 */
#include "../tensor.h"
#include "../training/intermediates.h"
#include <torch/torch.h>
#include <iostream>

extern "C" int param_count(void);
extern "C" void* param_tensor(int i);
extern "C" void param_clear(void);

extern "C" const char* backend_name(void) {
	return "torch";
}

extern "C" void backend_reset_for_eval(void) {
	free_intermediates();
	for (int i_ = 0; i_ < param_count(); i_++) {
		auto* tensor = (at::Tensor*)param_tensor(i_);
		if (tensor->grad().defined()) tensor->grad().zero_();
	}
}

/* See backend.h: explicit pre-exit cleanup of every persistent
 * at::Tensor*. Forces ~at::Tensor → ~Storage → CPUAllocator-free
 * cascades to run inside `main` rather than during process shutdown
 * (where on large-model CPU lanes the same work takes 14-22 minutes
 * via libtorch's per-tensor destructor cascade). Measured 2026-05-28
 * on HfLlama-1.2B BF16 torch-cpu: wall 23m22s → 1m21s, exit 0.
 *
 * Two phases:
 *   (1) free_intermediates() — bulk-deletes ~600 forward-pass
 *       intermediates in intermediates_torch (saves ~10 min).
 *   (2) walk param_registry_arr deleting each at::Tensor* —
 *       releases the ~146 params (saves the remaining ~13 min).
 * param_clear() resets the registry count; tensor_release_handle on
 * torch is a no-op so calling it on the now-freed pointer is safe
 * (pointer read but not dereferenced).
 *
 * **CPU-only**: on MPS/CUDA `delete (at::Tensor*)` forces a per-tensor
 * Metal/CUDA stream sync (~146 syncs of ~7 s each on a 1.2 B model)
 * which REGRESSED torch-mps wall 6:42 → 24:07. The async device
 * release that runs at process exit is fine on GPU lanes. We probe
 * the first param's device and bail when it's not CPU. */
extern "C" void backend_release_all_persistent(void) {
	int n = param_count();
	if (n > 0) {
		auto* first = (at::Tensor*)param_tensor(0);
		if (first && !first->is_cpu()) {
			/* GPU lane — async device release is cheap; explicit
			 * delete forces sync per tensor and regresses wall. */
			return;
		}
	}
	free_intermediates();
	for (int i_ = 0; i_ < n; i_++) {
		auto* tensor = (at::Tensor*)param_tensor(i_);
		delete tensor;
	}
	param_clear();
}

extern "C" void tensor_print(TensorHandle h) {
	// std::cout << at::Tensor requires the tensor to live on CPU.
	std::cout << to_tensor(h)->cpu() << std::endl;
}

/* mx::compile is mlx-only; torch backend always reports disabled
   regardless of MLX_COMPILE env var. */
extern "C" int tensor_mlx_compile_enabled(void) {
	return 0;
}
extern "C" int tensor_mlx_compile_invocations(void) {
	return 0;
}
extern "C" void tensor_mlx_compile_reset_stats(void) {
}

/* torch-only Criterion suite for the backend meta surface.
 *
 * Targets backend_meta.cpp paths the common tape suite never reaches on
 * the torch backend:
 *   - backend_name -> "torch".
 *   - backend_release_all_persistent CPU path: probes the first param's
 *     device (is_cpu() true on the CI lane), then bulk-deletes the
 *     param_registry at::Tensor*s and resets the registry count.
 *
 * The GPU-lane early return (!is_cpu()) is MPS/CUDA-only and excluded in
 * the product .cpp — the torch CI lane runs the CPU device.
 *
 * torch CPU base dtype is F64.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

Test(torch_core_backend_meta, backend_name_is_torch) {
	cr_assert_str_eq(backend_name(), "torch", "backend_name should be \"torch\" (got %s)",
	                 backend_name());
}

Test(torch_core_backend_meta, release_all_persistent_cpu_path) {
	/* Register a CPU param so backend_release_all_persistent enters the
	   n > 0 block, probes is_cpu() (true on CI), and walks the registry
	   deleting each at::Tensor* before param_clear() resets the count. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_f64(hcopy(xd, 3), shape, 1, /*requires_grad=*/1);
	param_register("p", x);
	cr_assert_eq(param_count(), 1, "registry should hold one param before release");
	backend_release_all_persistent();
	cr_assert_eq(param_count(), 0, "release_all_persistent should reset the registry count");
}

Test(torch_core_backend_meta, release_all_persistent_empty_registry) {
	/* n == 0 branch: skips the device probe, free_intermediates only,
	   loop body never runs, param_clear() leaves count at 0. */
	param_clear();
	cr_assert_eq(param_count(), 0, "registry should be empty");
	backend_release_all_persistent();
	cr_assert_eq(param_count(), 0, "release on empty registry should keep count 0");
}

#endif /* BACKEND_TORCH */

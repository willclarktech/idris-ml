/* torch-only Criterion suite for the CUDA device target.
 *
 * Colocated with device.cpp: exercises the "cuda" arm of
 * tensor_to_device / tensor_to_device_persistent that the CPU CI lane
 * never reaches. Every test starts from the EAFP probe the backend
 * already provides — tensor_to_device(h, "cuda") returns NULL when no
 * CUDA device is present (device.cpp catches the c10::Error) — and
 * SKIPs on NULL, so the suite stays green on macOS/CPU lanes and only
 * asserts on a CUDA box (Colab via scripts/test_cuda_colab.sh, or any
 * Linux host with a CUDA-enabled libtorch).
 *
 * Coverage mirrors the retired flat-csrc Colab probe: placement,
 * on-device arithmetic with CPU round-trip readback, and a backward
 * pass whose gradient lands on a CUDA-resident param.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* EAFP availability probe: migrate a throwaway tensor to "cuda".
   NULL = no CUDA device (or CPU-only libtorch wheel) -> caller skips. */
static TensorHandle cuda_probe(void) {
	double xd[] = {1.0};
	int shape[] = {1};
	TensorHandle x = tensor_create_f64(hcopy(xd, 1), shape, 1, /*requires_grad=*/0);
	return tensor_to_device(x, "cuda");
}

#define REQUIRE_CUDA()                                                                             \
	do {                                                                                           \
		if (cuda_probe() == NULL) cr_skip_test("no CUDA device available (EAFP probe -> NULL)");   \
	} while (0)

Test(torch_cuda, placement_reports_cuda_device) {
	/* tensor_to_device(h, "cuda"): the migrated handle must report a
	   cuda:* device string. */
	REQUIRE_CUDA();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	int shape[] = {4};
	TensorHandle x = tensor_create_f64(hcopy(xd, 4), shape, 1, /*requires_grad=*/0);
	TensorHandle g = tensor_to_device(x, "cuda");
	cr_assert_neq(g, NULL, "to_device(cuda) should succeed after positive probe");
	const char* dev = tensor_device(g);
	cr_assert_neq(dev, NULL, "tensor_device should return a non-NULL string");
	cr_assert(strstr(dev, "cuda") != NULL, "device string should contain \"cuda\" (got %s)", dev);
}

Test(torch_cuda, add_on_gpu_roundtrips_values) {
	/* On-device add, then migrate back to cpu and verify values. F64
	   whole/half numbers are exact through the round trip. */
	REQUIRE_CUDA();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	int shape[] = {4};
	TensorHandle x = tensor_create_f64(hcopy(xd, 4), shape, 1, /*requires_grad=*/0);
	TensorHandle g = tensor_to_device(x, "cuda");
	TensorHandle s = tensor_add(g, g);
	cr_assert_neq(s, NULL, "add on cuda tensors should succeed");
	cr_assert(strstr(tensor_device(s), "cuda") != NULL, "add result should stay on cuda (got %s)",
	          tensor_device(s));
	TensorHandle back = tensor_to_device(s, "cpu");
	cr_assert_neq(back, NULL, "migration back to cpu should succeed");
	double buf[4];
	tensor_to_doubles(back, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], 2.0 * xd[i], 1e-12,
		                   "cuda add readback [%d]: expected %.1f got %.9f", i, 2.0 * xd[i],
		                   buf[i]);
	}
}

Test(torch_cuda, backward_grad_lands_on_cuda_param) {
	/* d(w*x)/dw = x with both operands CUDA-resident. The param
	   migration uses the persistent variant (param lifetime; see
	   device.cpp), and the grad reads back through the registry. */
	REQUIRE_CUDA();
	param_clear();
	double wd[] = {2.0};
	double xd[] = {3.0};
	int shape[] = {1};
	TensorHandle w = tensor_to_device_persistent(
	    tensor_create_f64(hcopy(wd, 1), shape, 1, /*requires_grad=*/1), "cuda");
	cr_assert_neq(w, NULL, "persistent migration of the param should succeed");
	param_register("cuda_smoke.w", w);
	TensorHandle x =
	    tensor_to_device(tensor_create_f64(hcopy(xd, 1), shape, 1, /*requires_grad=*/0), "cuda");
	TensorHandle y = tensor_mul(w, x);
	cr_assert(strstr(tensor_device(y), "cuda") != NULL, "mul result should stay on cuda (got %s)",
	          tensor_device(y));
	tensor_backward(y);
	TensorHandle grad = tensor_grad(w);
	cr_assert_neq(grad, NULL, "w should carry a grad after backward");
	TensorHandle grad_cpu = tensor_to_device(grad, "cpu");
	cr_assert_float_eq(tensor_item(grad_cpu), 3.0, 1e-12, "d(w*x)/dw should be x=3.0 (got %.9f)",
	                   tensor_item(grad_cpu));
}

#endif /* BACKEND_TORCH */

/* torch-only Criterion suite for the dtype-cast + device surfaces.
 *
 * Targets two torch-specific files the common tape suite never reaches:
 *   - core/lifecycle/cast.cpp: tensor_cast_dtype_f32 / _f64. Both route
 *     through at::Tensor::to(dtype) + from_tensor (the tracking variant),
 *     so the cast is an autograd-traced intermediate. Drives the value +
 *     dtype-tag side of each, plus gradient flow through the F32->F64 cast.
 *   - device.cpp: tensor_to_device / tensor_to_device_persistent /
 *     tensor_device. The torch CI lane is CPU, so only the CPU/identity
 *     path is reachable here. The c10::Error catch branches (invalid
 *     device string) and the MPS/CUDA device targets are NOT exercised on
 *     the CPU lane (see uncertainties / EXCL notes for those branches).
 *
 * libtorch CPU is multi-dtype, so the bare F32 creators construct real
 * F32 storage. F32 readback asserts at 1e-5; F64 whole-number ints exact.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* Streamed-creator data args are callee-owned (freed inside), so every
   buffer handed to a creator is a fresh heap copy. The non-streamed
   _f32/_f64 creators copy too; heap-copy uniformly for safety. */

/* ---------------------------------------------------------------------
   cast.cpp — tensor_cast_dtype_f32 / _f64
   --------------------------------------------------------------------- */

Test(torch_core_lifecycle_cast, f64_to_f32_value_and_tag) {
	/* tensor_cast_dtype_f32: F64 source -> F32 storage, values preserved. */
	double xd[] = {1.5, -2.25, 3.75, 0.0};
	int shape[] = {2, 2};
	TensorHandle x = tensor_create_f64(hcopy(xd, 4), shape, 2, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "source should be F64 (got %s)",
	                 tensor_dtype_name(x));
	TensorHandle y = tensor_cast_dtype_f32(x);
	cr_assert_str_eq(tensor_dtype_name(y), "F32", "cast target should be F32 (got %s)",
	                 tensor_dtype_name(y));
	cr_assert_eq(tensor_numel(y), 4, "numel preserved through cast");
	cr_assert_eq(tensor_dim(y), 2, "rank preserved through cast");
	cr_assert_eq(tensor_size(y, 1), 2, "dim 1 preserved through cast");
	double buf[4];
	tensor_to_doubles(y, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 cast readback [%d]: expected %.3f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(torch_core_lifecycle_cast, f32_to_f64_value_and_tag) {
	/* tensor_cast_dtype_f64: F32 source -> F64 storage. Whole numbers are
	   exact in both, so the readback is bit-stable. */
	double xd[] = {2.0, -4.0, 8.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_f32(hcopy(xd, 3), shape, 1, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "source should be F32 (got %s)",
	                 tensor_dtype_name(x));
	TensorHandle y = tensor_cast_dtype_f64(x);
	cr_assert_str_eq(tensor_dtype_name(y), "F64", "cast target should be F64 (got %s)",
	                 tensor_dtype_name(y));
	double buf[3];
	tensor_to_doubles(y, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12, "F64 cast readback [%d]: expected %.1f got %.9f",
		                   i, xd[i], buf[i]);
	}
}

Test(torch_core_lifecycle_cast, grad_flows_through_cast) {
	/* Both source and target floating -> at::Tensor::to is autograd-traced.
	   A requires_grad F64 param cast to F32 still backprops grad=1 to the
	   registered source param. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_f64(hcopy(xd, 3), shape, 1, /*requires_grad=*/1);
	param_register("x", x);
	TensorHandle y = tensor_cast_dtype_f32(x);
	TensorHandle loss = tensor_sum(y);
	cr_assert_float_eq(tensor_item(loss), 6.0, 1e-5, "sum of cast should be 6 (got %.6f)",
	                   tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5,
		                   "grad through cast x[%d] should be 1 (got %.6f)", i,
		                   param_grad_item_at(0, i));
	}
}

/* ---------------------------------------------------------------------
   device.cpp — tensor_to_device / _persistent / tensor_device
   (CPU/identity path only; MPS/CUDA targets + catch branches not
   reachable on the CPU CI lane)
   --------------------------------------------------------------------- */

Test(torch_device, to_device_cpu_identity_preserves_values) {
	/* tensor_to_device(h, "cpu"): identity migration on the CPU lane.
	   Values + shape survive; result rides the intermediates vector. */
	double xd[] = {5.0, 6.0, 7.0, 8.0};
	int shape[] = {2, 2};
	TensorHandle x = tensor_create_f64(hcopy(xd, 4), shape, 2, /*requires_grad=*/0);
	TensorHandle y = tensor_to_device(x, "cpu");
	cr_assert_neq(y, NULL, "to_device(cpu) should succeed (non-NULL)");
	cr_assert_eq(tensor_numel(y), 4, "numel preserved through to_device");
	cr_assert_eq(tensor_dim(y), 2, "rank preserved through to_device");
	double buf[4];
	tensor_to_doubles(y, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12, "to_device readback [%d]: expected %.1f got %.9f",
		                   i, xd[i], buf[i]);
	}
}

Test(torch_device, to_device_persistent_cpu_preserves_values) {
	/* tensor_to_device_persistent(h, "cpu"): same migration but the result
	   is NOT pushed onto the intermediates vector (param-lifetime). */
	double xd[] = {-1.0, 0.0, 2.5};
	int shape[] = {3};
	TensorHandle x = tensor_create_f64(hcopy(xd, 3), shape, 1, /*requires_grad=*/0);
	TensorHandle y = tensor_to_device_persistent(x, "cpu");
	cr_assert_neq(y, NULL, "to_device_persistent(cpu) should succeed (non-NULL)");
	cr_assert_eq(tensor_numel(y), 3, "numel preserved through persistent migration");
	double buf[3];
	tensor_to_doubles(y, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12,
		                   "persistent migration readback [%d]: expected %.2f got %.9f", i, xd[i],
		                   buf[i]);
	}
}

Test(torch_device, tensor_device_reports_cpu) {
	/* tensor_device: a CPU-constructed tensor reports a "cpu" device str. */
	double xd[] = {1.0, 2.0};
	int shape[] = {2};
	TensorHandle x = tensor_create_f64(hcopy(xd, 2), shape, 1, /*requires_grad=*/0);
	const char* dev = tensor_device(x);
	cr_assert_neq(dev, NULL, "tensor_device should return a non-NULL string");
	cr_assert(strncmp(dev, "cpu", 3) == 0, "device string should start with \"cpu\" (got %s)", dev);
}

Test(torch_device, tensor_device_stable_after_cpu_roundtrip) {
	/* After an explicit to_device("cpu"), the device str is still cpu. */
	double xd[] = {3.0, 4.0, 5.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_f64(hcopy(xd, 3), shape, 1, /*requires_grad=*/0);
	TensorHandle y = tensor_to_device(x, "cpu");
	const char* dev = tensor_device(y);
	cr_assert(strncmp(dev, "cpu", 3) == 0, "post-roundtrip device should be cpu (got %s)", dev);
}

#endif /* BACKEND_TORCH */

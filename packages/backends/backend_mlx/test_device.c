/* mlx-only Criterion suite for the device surface (device.cpp).
 *
 * mlx has no torch-style intra-device migration: tensors live on the
 * stream they were created on, so `tensor_to_device` and its
 * param-lifetime sibling `tensor_to_device_persistent` are no-op
 * identities, and `tensor_device` returns the "gpu" placeholder.
 *
 * device.cpp showed 0% line coverage in the mlx baseline (the common
 * tape suite never drives the device FFI on the mlx lane). These tests
 * call each of the three exported functions and assert the documented
 * behavior: handle/value identity through the migration shims and the
 * "gpu" device string.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* The non-param 2d creator owns (frees) its host buffer, so feed it a
   heap copy rather than a stack array. */

/* tensor_to_device on mlx is identity: same handle back, value intact. */
Test(mlx_device_to_device, identity_preserves_handle_and_value) {
	TensorHandle x = tensor_create_scalar(3.5, /*requires_grad=*/0);
	TensorHandle y = tensor_to_device(x, "gpu");
	cr_assert_eq(y, x, "tensor_to_device should return the same handle (identity)");
	cr_assert_float_eq(tensor_item(y), 3.5, TEST_TOL_TIGHT,
	                   "value should survive the identity migration");
}

/* The device-string argument is ignored — any string yields identity. */
Test(mlx_device_to_device, ignores_device_string) {
	TensorHandle x = tensor_create_scalar(-2.0, /*requires_grad=*/0);
	TensorHandle y = tensor_to_device(x, "cpu");
	cr_assert_eq(y, x, "tensor_to_device must ignore the device arg and return identity");
	cr_assert_float_eq(tensor_item(y), -2.0, TEST_TOL_TIGHT, "value preserved with 'cpu' arg too");
}

/* tensor_to_device_persistent shares the identity implementation. */
Test(mlx_device_to_device_persistent, identity_preserves_handle_and_value) {
	TensorHandle x = tensor_create_scalar(7.25, /*requires_grad=*/1);
	TensorHandle y = tensor_to_device_persistent(x, "gpu");
	cr_assert_eq(y, x, "tensor_to_device_persistent should return the same handle (identity)");
	cr_assert_float_eq(tensor_item(y), 7.25, TEST_TOL_TIGHT,
	                   "value should survive the persistent identity migration");
}

/* tensor_device returns the "gpu" placeholder string regardless of tensor. */
Test(mlx_device_query, returns_gpu_placeholder) {
	TensorHandle x = tensor_create_scalar(1.0, /*requires_grad=*/0);
	const char* dev = tensor_device(x);
	cr_assert_not_null(dev, "tensor_device must not return NULL");
	cr_assert_str_eq(dev, "gpu", "mlx tensor_device should return the 'gpu' placeholder (got %s)",
	                 dev);
}

/* The device string is independent of the tensor handle (arg ignored). */
Test(mlx_device_query, gpu_for_multidim_tensor) {
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_2d_f32(2, 2, hcopy(xd, 4), /*requires_grad=*/0);
	cr_assert_str_eq(tensor_device(x), "gpu",
	                 "device string should be 'gpu' for a 2x2 tensor as well");
}

#endif /* BACKEND_MLX */

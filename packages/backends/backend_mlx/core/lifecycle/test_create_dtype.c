/* mlx-only Criterion suite for the multi-rank tensor_create_* dtype path.
 *
 * Targets the per-dtype base creators in core/lifecycle/create.cpp that
 * the common tape suite never reaches:
 *   - tensor_create_f32 / tensor_create_f64 (public unsuffixed F32/F64).
 *   - tensor_create_streamed dtag dispatch into the BF16 (17), F16 (13)
 *     and I32 (10) base creators (the data/shape/rank signatures).
 *
 * Each creator routes through mx_array_from_doubles + tensor_create_impl;
 * this drives the value/shape/dtype-tag side of that helper for every
 * storage dtype mlx admits. BF16/F16 carry reduced mantissa, so the
 * value asserts use whole numbers (exact in every float dtype) at a
 * relaxed tolerance; F32/F64 assert at 1e-5.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* DType.Core dtag values (kind-major layout): 10=I32, 13=F16, 14=F32,
   15=F64, 17=BF16. */
#define DTAG_I32 10
#define DTAG_F16 13
#define DTAG_BF16 17

Test(mlx_core_lifecycle_create_dtype, f32_public_roundtrip) {
	/* tensor_create_f32: public unsuffixed F32 multi-rank creator. */
	double xd[] = {1.5, -2.25, 3.75, 0.0};
	int shape[] = {2, 2};
	TensorHandle x = tensor_create_f32(hcopy(xd, 4), shape, 2, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "tensor_create_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 4, "numel should be 4");
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 readback [%d]: expected %.3f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_create_dtype, f64_public_roundtrip) {
	/* tensor_create_f64: public unsuffixed F64 multi-rank creator. */
	double xd[] = {10.0, -20.0, 30.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_f64(hcopy(xd, 3), shape, 1, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "tensor_create_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F64 readback [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_create_dtype, f32_requires_grad_appends_const) {
	/* requires_grad != 0 branch of tensor_create_impl: OP_CONST tape leaf,
	   then sum->backward gives elementwise grad of 1. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_f32(hcopy(xd, 3), shape, 1, /*requires_grad=*/1);
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	cr_assert_float_eq(tensor_item(loss), 12.0, 1e-5, "sum should be 12 (got %.6f)",
	                   tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5, "grad x[%d] should be 1 (got %.6f)",
		                   i, param_grad_item_at(0, i));
	}
}

Test(mlx_core_lifecycle_create_dtype, streamed_bf16_dtag) {
	/* tensor_create_streamed dtag=17 -> tensor_create_bf16_mlx_streamed. */
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	int shape[] = {4};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 4), shape, 1, /*requires_grad=*/0,
	                                        /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "dtag=17 should yield BF16 storage (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		/* Powers of two are exact in bf16. */
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "BF16 readback [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_create_dtype, streamed_f16_dtag) {
	/* tensor_create_streamed dtag=13 -> tensor_create_f16_mlx_streamed. */
	double xd[] = {3.0, -5.0, 16.0};
	int shape[] = {3};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 3), shape, 1, /*requires_grad=*/0,
	                                        /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "dtag=13 should yield F16 storage (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		/* Small integers are exact in f16. */
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "F16 readback [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_create_dtype, streamed_i32_dtag) {
	/* tensor_create_streamed dtag=10 -> tensor_create_i32_mlx_streamed. */
	double xd[] = {7.0, -8.0, 100.0, 0.0};
	int shape[] = {2, 2};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 4), shape, 2, /*requires_grad=*/0,
	                                        /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "dtag=10 should yield I32 storage (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "I32 readback [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

#endif /* BACKEND_MLX */

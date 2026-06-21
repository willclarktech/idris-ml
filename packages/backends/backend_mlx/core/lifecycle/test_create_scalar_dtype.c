/* mlx-only Criterion suite for the scalar tensor_create_scalar_* dtype path.
 *
 * Targets the per-dtype scalar creators in core/lifecycle/create_scalar.cpp
 * the common tape suite never reaches:
 *   - tensor_create_scalar_f32 / tensor_create_scalar_f64 (public F32/F64).
 *   - tensor_create_scalar_streamed dtag dispatch into the F16 (13) and
 *     I32 (10) base scalar creators.
 *
 * Each routes through tensor_create_scalar_impl (mx::array(value, dt)).
 * BF16/F16 carry reduced mantissa, so the F16 case uses a whole number
 * (exact) at relaxed tolerance; F32/F64 assert at 1e-5; I32 is exact.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* DType.Core dtag values: 10=I32, 13=F16, 14=F32, 15=F64, 17=BF16. */
#define DTAG_I32 10
#define DTAG_F16 13

Test(mlx_core_lifecycle_create_scalar_dtype, f32_public_value) {
	TensorHandle x = tensor_create_scalar_f32(2.5, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F32",
	                 "tensor_create_scalar_f32 should yield F32 (got %s)", tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 1, "scalar numel should be 1");
	cr_assert_float_eq(tensor_item(x), 2.5, 1e-5, "scalar value should be 2.5 (got %.9f)",
	                   tensor_item(x));
}

Test(mlx_core_lifecycle_create_scalar_dtype, f64_public_value) {
	TensorHandle x = tensor_create_scalar_f64(-7.125, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F64",
	                 "tensor_create_scalar_f64 should yield F64 (got %s)", tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), -7.125, 1e-5, "scalar value should be -7.125 (got %.9f)",
	                   tensor_item(x));
}

Test(mlx_core_lifecycle_create_scalar_dtype, f32_requires_grad_appends_const) {
	/* requires_grad branch: OP_CONST leaf; sum(scalar)=value, grad=1. */
	param_clear();
	TensorHandle x = tensor_create_scalar_f32(5.0, /*requires_grad=*/1);
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	cr_assert_float_eq(tensor_item(loss), 5.0, 1e-5, "sum should be 5 (got %.6f)",
	                   tensor_item(loss));
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-5, "scalar grad should be 1 (got %.6f)",
	                   param_grad_item_at(0, 0));
}

Test(mlx_core_lifecycle_create_scalar_dtype, streamed_f16_dtag) {
	/* tensor_create_scalar_streamed dtag=13 -> scalar_f16 base creator. */
	TensorHandle x = tensor_create_scalar_streamed(8.0, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "scalar dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 8.0, TEST_TOL_RELAXED,
	                   "F16 scalar value should be 8 (got %.9f)", tensor_item(x));
}

Test(mlx_core_lifecycle_create_scalar_dtype, streamed_i32_dtag) {
	/* tensor_create_scalar_streamed dtag=10 -> scalar_i32 base creator. */
	TensorHandle x = tensor_create_scalar_streamed(-42.0, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "scalar dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), -42.0, 0.0, "I32 scalar value should be -42 (got %.6f)",
	                   tensor_item(x));
}

#endif /* BACKEND_MLX */

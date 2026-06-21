/* mlx-only Criterion suite for tensor_cast_dtype_{f32,f64} (forward + backward).
 *
 *   tensor_cast_dtype_f32(h):  mx::astype to mx::float32
 *   tensor_cast_dtype_f64(h):  mx::astype to mx::float64
 *
 * Backward: cast records OP_CAST_DTYPE with scalar_arg encoding the
 * target (0.0 = F32, 1.0 = F64); replay re-runs mx::astype so the
 * gradient flows through unchanged.
 *
 * Closes the W4 OP_CAST_DTYPE coverage gap on mlx. This file lives
 * under test/mlx/ (not test/common/) because the symmetric F32<->F64
 * cast surface is mlx-only: tape's F32 storage is a separate codepath
 * (tape_cast_dtype_*) and torch's libtorch backend does its own cast
 * via .to(opts) without the OP_CAST_DTYPE replay tag.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

static double* heap_copy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

Test(mlx_core_lifecycle_cast_dtype, f64_to_f32_round_trip) {
	/* Default mlx tensor is F32 (per project_mlx_gpu_environment.md);
	 * cast to F64 then back to F32 should preserve values within F32
	 * precision (since the original was already F32-representable). */
	param_clear();
	double xd[] = {1.0, 2.5, -3.75, 0.125};
	TensorHandle x = tensor_create_2d_f64(2, 2, heap_copy(xd, 4), 0);
	/* x defaults to F32 storage on mlx; cast to F64 then back. */
	TensorHandle as_f64 = tensor_cast_dtype_f64(x);
	cr_assert_str_eq(tensor_dtype_name(as_f64), "F64",
	                 "after cast_dtype_f64, dtype should be F64 (got %s)",
	                 tensor_dtype_name(as_f64));
	TensorHandle as_f32_again = tensor_cast_dtype_f32(as_f64);
	cr_assert_str_eq(tensor_dtype_name(as_f32_again), "F32",
	                 "after cast_dtype_f32, dtype should be F32 (got %s)",
	                 tensor_dtype_name(as_f32_again));
	double buf[4];
	tensor_to_doubles(as_f32_again, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "round-trip buf[%d] should match input %.3f (got %.9f)", i, xd[i],
		                   buf[i]);
	}
}

Test(mlx_core_lifecycle_cast_dtype, backward_passes_gradient) {
	/* Forward: y = cast_dtype_f64(x); loss = sum(y); dL/dx = 1 elementwise. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	TensorHandle x = tensor_create_param_2d_f64(1, 3, heap_copy(xd, 3));
	param_register("x", x);
	TensorHandle y = tensor_cast_dtype_f64(x);
	TensorHandle loss = tensor_sum(y);
	cr_assert_float_eq(tensor_item(loss), 6.0, TEST_TOL_RELAXED,
	                   "sum after cast should be 6 (got %.9f)", tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED,
		                   "grad x[%d] should pass through as 1 (got %.9f)", i,
		                   param_grad_item_at(0, i));
	}
}

Test(mlx_core_lifecycle_cast_dtype, f32_to_f16_forward) {
	/* tensor_cast_dtype_f16_mlx_streamed via the public unified dispatch
	   (dtag=13). f16 has 11 mantissa bits; powers of two are exact. */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, -8.0};
	TensorHandle x = tensor_create_2d_f64(2, 2, heap_copy(xd, 4), 0);
	TensorHandle as_f16 = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, /*dtag=*/13);
	cr_assert_str_eq(tensor_dtype_name(as_f16), "F16",
	                 "after cast_dtype f16, dtype should be F16 (got %s)",
	                 tensor_dtype_name(as_f16));
	double buf[4];
	tensor_to_doubles(as_f16, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "f16 cast buf[%d] expected %.3f got %.9f", i, xd[i],
		                   buf[i]);
	}
}

Test(mlx_core_lifecycle_cast_dtype, f32_to_i32_forward) {
	/* tensor_cast_dtype_i32_mlx_streamed via the public unified dispatch
	   (dtag=10). astype to int32 truncates toward zero; whole-number
	   inputs are exact. */
	param_clear();
	double xd[] = {1.0, -2.0, 7.0, -42.0};
	TensorHandle x = tensor_create_2d_f64(2, 2, heap_copy(xd, 4), 0);
	TensorHandle as_i32 = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, /*dtag=*/10);
	cr_assert_str_eq(tensor_dtype_name(as_i32), "I32",
	                 "after cast_dtype i32, dtype should be I32 (got %s)",
	                 tensor_dtype_name(as_i32));
	double buf[4];
	tensor_to_doubles(as_i32, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "i32 cast buf[%d] expected %.0f got %.6f", i, xd[i],
		                   buf[i]);
	}
}

Test(mlx_core_lifecycle_cast_dtype, backward_replay_bf16) {
	/* Forward: y = cast_bf16(x); loss = sum(y); dL/dx = 1 elementwise.
	   Exercises the OP_CAST_DTYPE replay case 2 (bf16, lines 73-74). */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_param_2d_f64(1, 3, heap_copy(xd, 3));
	param_register("x", x);
	TensorHandle y = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, /*dtag=*/17);
	cr_assert_str_eq(tensor_dtype_name(y), "BF16", "cast target should be BF16 (got %s)",
	                 tensor_dtype_name(y));
	TensorHandle loss = tensor_sum(y);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 0.05,
		                   "grad x[%d] should pass through bf16 cast as 1 (got %.9f)", i,
		                   param_grad_item_at(0, i));
	}
}

Test(mlx_core_lifecycle_cast_dtype, backward_replay_f16) {
	/* Forward: y = cast_f16(x); loss = sum(y); dL/dx = 1 elementwise.
	   Exercises the OP_CAST_DTYPE replay case 3 (f16, lines 76-77). */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_param_2d_f64(1, 3, heap_copy(xd, 3));
	param_register("x", x);
	TensorHandle y = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, /*dtag=*/13);
	cr_assert_str_eq(tensor_dtype_name(y), "F16", "cast target should be F16 (got %s)",
	                 tensor_dtype_name(y));
	TensorHandle loss = tensor_sum(y);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 0.01,
		                   "grad x[%d] should pass through f16 cast as 1 (got %.9f)", i,
		                   param_grad_item_at(0, i));
	}
}

#endif /* BACKEND_MLX */

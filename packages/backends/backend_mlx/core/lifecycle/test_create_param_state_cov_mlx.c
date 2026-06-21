/* mlx-only Criterion coverage suite for the persistent-leaf creators —
 * the float-dtag edge arms test_create_param_state.c does NOT reach.
 *
 * Targets core/lifecycle/create_param_state.cpp. The sibling suite covers
 * the public F32/F64 entry points plus a scattered subset of the per-dtype
 * _mlx_streamed arms (e.g. param_1d bf16/f16, param_2d i32/bf16). This file
 * closes the remaining dtag arms on each rank:
 *   - the plain non-grad 1d/2d creators (tensor_create_{1,2}d_*_mlx_streamed)
 *     driven via the tensor_create_{1,2}d_streamed dispatchers across
 *     BF16 (17) / F16 (13) / I32 (10);
 *   - the param creators param_1d i32, param_2d f16, param_3d bf16,
 *     param_4d f16;
 *   - the state creators state_1d f16, state_2d bf16.
 *
 * Each creator routes through tensor_create_impl + mx_array_from_doubles.
 * BF16/F16 carry reduced mantissa, so value asserts use whole numbers /
 * powers of two (exact in every float dtype) at TEST_TOL_RELAXED; I32 is
 * exact (0.0 tol). These tests only read back value/shape/dtype — they do
 * not register params, so no param_clear is needed (mirrors the sibling's
 * param_*_public readback tests).
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
#define DTAG_F32 14
#define DTAG_F64 15
#define DTAG_BF16 17

/* ---------- Plain non-grad 1d creators (requires_grad threaded through) ---------- */

Test(create_param_state_cov, create_1d_streamed_bf16) {
	/* tensor_create_1d_streamed dtag=17 -> tensor_create_1d_bf16_mlx_streamed
	   -> tensor_create_1d_impl(float32->bfloat16). Powers of two exact. */
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x = tensor_create_1d_streamed(4, hcopy(xd, 4), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "1d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 1, "rank should be 1");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "1d BF16 [%d]: expected %.1f got %.9f",
		                   i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, create_1d_streamed_f16) {
	/* tensor_create_1d_streamed dtag=13 -> tensor_create_1d_f16_mlx_streamed. */
	double xd[] = {3.0, -5.0, 16.0};
	TensorHandle x =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*requires_grad=*/0, /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "1d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "1d F16 [%d]: expected %.1f got %.9f",
		                   i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, create_1d_streamed_i32) {
	/* tensor_create_1d_streamed dtag=10 -> tensor_create_1d_i32_mlx_streamed. */
	double xd[] = {7.0, -8.0, 100.0, 0.0};
	TensorHandle x =
	    tensor_create_1d_streamed(4, hcopy(xd, 4), /*requires_grad=*/0, /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "1d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "1d I32 [%d]: expected %.0f got %.6f", i, xd[i],
		                   buf[i]);
	}
}

/* ---------- Plain non-grad 2d creators ---------- */

Test(create_param_state_cov, create_2d_streamed_bf16) {
	/* tensor_create_2d_streamed dtag=17 -> tensor_create_2d_bf16_mlx_streamed. */
	double xd[] = {2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
	TensorHandle x = tensor_create_2d_streamed(2, 3, hcopy(xd, 6), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "2d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	cr_assert_eq(tensor_size(x, 1), 3, "dim 1 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "2d BF16 [%d]: expected %.1f got %.9f",
		                   i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, create_2d_streamed_f16) {
	/* tensor_create_2d_streamed dtag=13 -> tensor_create_2d_f16_mlx_streamed. */
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "2d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "2d F16 [%d]: expected %.1f got %.9f",
		                   i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, create_2d_streamed_i32) {
	/* tensor_create_2d_streamed dtag=10 -> tensor_create_2d_i32_mlx_streamed. */
	double xd[] = {11.0, -12.0, 13.0, -14.0, 15.0, -16.0};
	TensorHandle x = tensor_create_2d_streamed(3, 2, hcopy(xd, 6), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "2d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "2d I32 [%d]: expected %.0f got %.6f", i, xd[i],
		                   buf[i]);
	}
}

/* ---------- Param creators: remaining dtag arms per rank ---------- */

Test(create_param_state_cov, param_1d_streamed_i32) {
	/* tensor_create_param_1d_streamed dtag=10 -> _i32_mlx_streamed. */
	double xd[] = {21.0, -22.0, 23.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param_1d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param_1d I32 [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

Test(create_param_state_cov, param_2d_streamed_f16) {
	/* tensor_create_param_2d_streamed dtag=13 -> _f16_mlx_streamed. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 3, hcopy(xd, 6), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param_2d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_2d F16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, param_3d_streamed_bf16) {
	/* tensor_create_param_3d_streamed dtag=17 -> _bf16_mlx_streamed. */
	double xd[] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param_3d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_3d BF16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, param_4d_streamed_f16) {
	/* tensor_create_param_4d_streamed dtag=13 -> _f16_mlx_streamed. */
	double xd[] = {1.0, -2.0, 3.0, -4.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param_4d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_4d F16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

/* ---------- State creators: remaining dtag arms ---------- */

Test(create_param_state_cov, state_1d_streamed_f16) {
	/* tensor_create_state_1d_streamed dtag=13 -> _f16_mlx_streamed (no tape leaf). */
	double xd[] = {2.0, 4.0, 6.0, 8.0};
	TensorHandle x = tensor_create_state_1d_streamed(4, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "state_1d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "state_1d F16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(create_param_state_cov, state_2d_streamed_bf16) {
	/* tensor_create_state_2d_streamed dtag=17 -> _bf16_mlx_streamed. */
	double xd[] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 3, hcopy(xd, 6), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "state_2d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	cr_assert_eq(tensor_size(x, 1), 3, "dim 1 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "state_2d BF16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

#endif /* BACKEND_MLX */

/* mlx-only Criterion suite for the persistent-leaf creators.
 *
 * Targets core/lifecycle/create_param_state.cpp — the param (1d/2d/3d/4d)
 * and state (1d/2d) creators across every storage dtype mlx admits, plus
 * the public unsuffixed F32/F64 convenience entry points. The common tape
 * suite never reaches these because tape rejects the bare F32 creators and
 * the multi-rank param/state path is driven only through the streamed
 * dispatchers in training/dtype_dispatch.cpp.
 *
 * Each creator routes data through tensor_create_impl + mx_array_from_doubles;
 * the param creators force requires_grad=1 (so they push an OP_CONST leaf and
 * participate in backward), the state creators force requires_grad=0
 * (persistent leaf, no tape node). The dtag values feed the per-dtype
 * _mlx_streamed creators via tensor_create_param_{1..4}d_streamed /
 * tensor_create_state_{1,2}d_streamed.
 *
 * BF16/F16 carry reduced mantissa, so their value asserts use whole numbers
 * (exact in every float dtype) at TEST_TOL_RELAXED; F32/F64 assert at 1e-5;
 * I32 is exact (0.0).
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

/* ---------- Public unsuffixed F32 / F64 convenience creators ---------- */

Test(mlx_core_lifecycle_param_state, create_1d_f32_public) {
	/* tensor_create_1d_f32 -> _mlx_streamed(default_stream) -> impl(float32). */
	double xd[] = {1.5, -2.25, 3.75};
	TensorHandle x = tensor_create_1d_f32(3, hcopy(xd, 3), /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "tensor_create_1d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 3, "numel should be 3");
	cr_assert_eq(tensor_dim(x), 1, "rank should be 1");
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 1d readback [%d]: expected %.3f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, create_1d_f64_public) {
	/* tensor_create_1d_f64 -> _mlx_streamed(default_stream) -> impl(float64). */
	double xd[] = {10.0, -20.0, 30.5, 0.0};
	TensorHandle x = tensor_create_1d_f64(4, hcopy(xd, 4), /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "tensor_create_1d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F64 1d readback [%d]: expected %.3f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, create_2d_f32_public) {
	/* tensor_create_2d_f32 -> _mlx_streamed -> tensor_create_2d_impl(float32). */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x = tensor_create_2d_f32(2, 3, hcopy(xd, 6), /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "tensor_create_2d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	cr_assert_eq(tensor_size(x, 1), 3, "dim 1 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 2d readback [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, create_2d_f64_default_public) {
	/* tensor_create_2d (unsuffixed) -> tensor_create_2d_f64 -> impl(float64). */
	double xd[] = {-1.5, 2.5, -3.5, 4.5};
	TensorHandle x = mk2d(2, 2, xd, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "tensor_create_2d should default to F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F64 2d readback [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

/* ---------- Public param 1d/2d F32/F64 + grad flow ---------- */

Test(mlx_core_lifecycle_param_state, param_1d_f32_grad) {
	/* tensor_create_param_1d_f32 forces requires_grad=1: sum->backward gives
	   elementwise grad of 1, proving the OP_CONST leaf is on the tape. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0};
	TensorHandle x = tensor_create_param_1d_f32(3, hcopy(xd, 3));
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param_1d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
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

Test(mlx_core_lifecycle_param_state, param_1d_f64_grad) {
	param_clear();
	double xd[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle x = tensor_create_param_1d_f64(4, hcopy(xd, 4));
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param_1d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	cr_assert_float_eq(tensor_item(loss), 4.0, 1e-5, "sum should be 4 (got %.6f)",
	                   tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5, "grad x[%d] should be 1 (got %.6f)",
		                   i, param_grad_item_at(0, i));
	}
}

Test(mlx_core_lifecycle_param_state, param_2d_f32_public) {
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_param_2d_f32(2, 2, hcopy(xd, 4));
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param_2d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_2d_f32 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_2d_f64_public) {
	double xd[] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
	TensorHandle x = tensor_create_param_2d_f64(3, 2, hcopy(xd, 6));
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param_2d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_size(x, 0), 3, "dim 0 should be 3");
	cr_assert_eq(tensor_size(x, 1), 2, "dim 1 should be 2");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_2d_f64 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

/* ---------- Public param 3d/4d F32/F64 ---------- */

Test(mlx_core_lifecycle_param_state, param_3d_f32_public) {
	/* 2x2x2 = 8 elements; tensor_create_param_3d_f32 -> impl(float32). */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x = tensor_create_param_3d_f32(2, 2, 2, hcopy(xd, 8));
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param_3d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_3d_f32 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_3d_f64_public) {
	double xd[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
	TensorHandle x = tensor_create_param_3d_f64(1, 2, 3, hcopy(xd, 6));
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param_3d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_size(x, 1), 2, "dim 1 should be 2");
	cr_assert_eq(tensor_size(x, 2), 3, "dim 2 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_3d_f64 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_4d_f32_public) {
	/* 1x1x2x2 = 4 elements; tensor_create_param_4d_f32 -> impl(float32). */
	double xd[] = {9.0, -8.0, 7.0, -6.0};
	TensorHandle x = tensor_create_param_4d_f32(1, 1, 2, 2, hcopy(xd, 4));
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param_4d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_4d_f32 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_4d_f64_public) {
	double xd[] = {1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25};
	TensorHandle x = tensor_create_param_4d_f64(2, 1, 2, 2, hcopy(xd, 8));
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param_4d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_4d_f64 [%d]: expected %.2f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

/* ---------- Public state 1d/2d F32/F64 (requires_grad=0) ---------- */

Test(mlx_core_lifecycle_param_state, state_1d_f32_no_grad) {
	/* state creators force requires_grad=0: no tape leaf, but value/shape
	   readback must still be correct. */
	double xd[] = {3.0, 6.0, 9.0};
	TensorHandle x = tensor_create_state_1d_f32(3, hcopy(xd, 3));
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "state_1d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "state_1d_f32 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, state_1d_f64_no_grad) {
	double xd[] = {-1.0, 0.0, 1.0, 2.0};
	TensorHandle x = tensor_create_state_1d_f64(4, hcopy(xd, 4));
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "state_1d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "state_1d_f64 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, state_2d_f32_no_grad) {
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x = tensor_create_state_2d_f32(3, 2, hcopy(xd, 6));
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "state_2d_f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "state_2d_f32 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, state_2d_f64_no_grad) {
	double xd[] = {0.0, -1.5, 2.5, -3.5};
	TensorHandle x = tensor_create_state_2d_f64(2, 2, hcopy(xd, 4));
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "state_2d_f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "state_2d_f64 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

/* ---------- Streamed param dispatch: BF16 / F16 / I32 across ranks ---------- */

Test(mlx_core_lifecycle_param_state, param_1d_streamed_bf16) {
	/* tensor_create_param_1d_streamed dtag=17 -> _bf16_mlx_streamed. Powers of
	   two are exact in bf16. */
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x = tensor_create_param_1d_streamed(4, hcopy(xd, 4), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param_1d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_1d BF16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_1d_streamed_f16) {
	double xd[] = {3.0, -5.0, 16.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param_1d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_1d F16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_2d_streamed_i32) {
	double xd[] = {7.0, -8.0, 9.0, -10.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param_2d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param_2d I32 [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_2d_streamed_bf16) {
	double xd[] = {2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 3, hcopy(xd, 6), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param_2d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_2d BF16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_3d_streamed_f16) {
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param_3d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_3d F16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_3d_streamed_i32) {
	double xd[] = {10.0, -20.0, 30.0, -40.0, 50.0, -60.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(1, 2, 3, hcopy(xd, 6), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param_3d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param_3d I32 [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_4d_streamed_bf16) {
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param_4d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "param_4d BF16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_4d_streamed_i32) {
	double xd[] = {3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(2, 1, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param_4d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param_4d I32 [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, param_4d_streamed_f32) {
	/* dtag=14 routes to _f32_mlx_streamed even via the streamed dispatcher. */
	double xd[] = {0.5, 1.5, 2.5, 3.5};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 1, 4, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param_4d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "param_4d F32 [%d]: expected %.1f got %.9f", i,
		                   xd[i], buf[i]);
	}
}

/* ---------- Streamed state dispatch: BF16 / F16 / I32 ---------- */

Test(mlx_core_lifecycle_param_state, state_1d_streamed_bf16) {
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "state_1d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "state_1d BF16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, state_1d_streamed_i32) {
	double xd[] = {11.0, -12.0, 13.0, -14.0};
	TensorHandle x = tensor_create_state_1d_streamed(4, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "state_1d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "state_1d I32 [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, state_2d_streamed_f16) {
	double xd[] = {2.0, 4.0, 6.0, 8.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "state_2d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "state_2d F16 [%d]: expected %.1f got %.9f", i, xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_param_state, state_2d_streamed_i32) {
	double xd[] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(3, 2, hcopy(xd, 6), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "state_2d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_size(x, 0), 3, "dim 0 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "state_2d I32 [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

/* ---- create_param_state_cov: remaining float-dtag arms (was test_create_param_state_cov_mlx.c)
 * ---- */

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

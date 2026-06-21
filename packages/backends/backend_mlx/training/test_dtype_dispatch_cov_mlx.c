/* mlx-only Criterion suite — coverage top-up for dtype-dispatch creators.
 *
 * Companion to test_dtype_dispatch.c. That suite leaves a handful of
 * FLOAT dtag case bodies in training/dtype_dispatch.cpp unexercised; this
 * file closes them so every f32 (dtag=14) / f64 (dtag=15) / bf16 (dtag=17)
 * / f16 (dtag=13) arm of each dispatcher runs at least once. The integer
 * (i32, dtag=10) routes are already covered by the companion suite and
 * core/lifecycle; the default-case abort arms (other int* + bool) are
 * principled exclusions and are not touched here.
 *
 * Specifically the float arms left uncovered before this file:
 *   - tensor_create_scalar_streamed      : bf16
 *   - tensor_create_streamed (rank-N)    : f32
 *   - tensor_create_1d_streamed          : bf16
 *   - tensor_create_2d_streamed          : f32
 *   - tensor_create_param_1d_streamed    : f32
 *   - tensor_create_param_2d_streamed    : bf16, f16
 *   - tensor_create_param_3d_streamed    : bf16, f16
 *   - tensor_create_param_4d_streamed    : bf16, f16
 *   - tensor_create_state_1d_streamed    : bf16, f16
 *   - tensor_create_state_2d_streamed    : bf16, f16
 *
 * Each test asserts the routed dtype name (a misroute would land in
 * mlx_dtype_unsupported -> abort) and reads the storage back. mlx is
 * F32-native; f32 roundtrips of small magnitudes are tight, f64-tagged
 * storage carries F32 precision (TEST_TOL_RELAXED). bf16/f16 inputs use
 * exactly-representable powers of two so value asserts stay simple.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* Idris-side dtag values mirroring DType.Core. */
#define DTAG_F16 13
#define DTAG_F32 14
#define DTAG_F64 15
#define DTAG_BF16 17

/* ---------------------------------------------------------------------
 * tensor_create_scalar_streamed — bf16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, scalar_bf16) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(8.0, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "scalar dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 8.0, 0.05, "scalar bf16 value (got %.6f)", tensor_item(x));
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_streamed (rank-N via shape/rank) — f32
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, rankn_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {2, 3};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 6), shape, /*rank=*/2,
	                                        /*requires_grad=*/0, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "rankN dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 6, "rankN f32 numel should be 6 (got %d)", tensor_numel(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "rankN f32 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_1d_streamed — bf16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, vec1d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(xd, 3), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "1d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(tensor_item_1d(x, i), xd[i], 0.05, "1d bf16 [%d] (got %.9f)", i,
		                   tensor_item_1d(x, i));
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_2d_streamed — f32
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, mat2d_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "2d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "2d f32 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_param_1d_streamed — f32
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, param1d_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param1d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "param1d f32 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_param_2d_streamed — bf16 / f16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, param2d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param2d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "param2d bf16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

Test(dtype_dispatch_cov, param2d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param2d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "param2d f16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_param_3d_streamed — bf16 / f16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, param3d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param3d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "param3d should be rank-3 (got %d)", tensor_dim(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "param3d bf16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

Test(dtype_dispatch_cov, param3d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param3d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "param3d f16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_param_4d_streamed — bf16 / f16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, param4d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param4d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "param4d should be rank-4 (got %d)", tensor_dim(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "param4d bf16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

Test(dtype_dispatch_cov, param4d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param4d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "param4d f16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_state_1d_streamed — bf16 / f16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, state1d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "state1d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "state1d bf16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

Test(dtype_dispatch_cov, state1d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "state1d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "state1d f16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

/* ---------------------------------------------------------------------
 * tensor_create_state_2d_streamed — bf16 / f16
 * ------------------------------------------------------------------- */

Test(dtype_dispatch_cov, state2d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "state2d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "state2d bf16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

Test(dtype_dispatch_cov, state2d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "state2d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "state2d f16 [%d] (got %.9f)", i, buf[i]);
	}
	param_clear();
}

#endif /* BACKEND_MLX */

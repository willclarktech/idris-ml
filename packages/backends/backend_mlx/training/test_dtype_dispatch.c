/* mlx-only Criterion suite for the dtype-dispatch creators.
 *
 * Drives every `tensor_create_*_streamed(... dtag)` dispatcher in
 * training/dtype_dispatch.cpp across the F32 (dtag=14), F64 (dtag=15)
 * and I32 lingua-franca (dtag=10) branches, plus the bf16 (17) / f16
 * (13) rungs where they fall on otherwise-uncovered case bodies. Each
 * test asserts the resulting dtype name and reads the values back so
 * the routed base creator actually ran (a wrong route would land in
 * mlx_dtype_unsupported -> abort, or misread the storage bits).
 *
 * The scalar / param-2d / param-3d / param-4d / state-1d / state-2d
 * dispatchers had zero coverage before this file (the common tape
 * suite never exercises the *_streamed dtag overloads, and the
 * existing core/lifecycle i32 + cast tests only touch 1d/2d/param-1d).
 *
 * mlx is F32-native; F64-tagged storage on the CPU stream carries F32
 * precision, so value asserts use TEST_TOL_RELAXED. Whole-number I32
 * casts are exact (tol 0.0).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* Idris-side dtag values mirroring DType.Core. */
#define DTAG_I32 10
#define DTAG_F16 13
#define DTAG_F32 14
#define DTAG_F64 15
#define DTAG_BF16 17

/* ---------------------------------------------------------------------
 * tensor_create_scalar_streamed — f32 / f64 / i32 / f16
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, scalar_f32) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(2.5, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "scalar dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 2.5, TEST_TOL_TIGHT, "scalar f32 value (got %.9f)",
	                   tensor_item(x));
}

Test(mlx_training_dtype_dispatch, scalar_f64) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(-3.25, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "scalar dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), -3.25, TEST_TOL_RELAXED, "scalar f64 value (got %.9f)",
	                   tensor_item(x));
}

Test(mlx_training_dtype_dispatch, scalar_i32) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(7.0, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "scalar dtag=10 should be I32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 7.0, 0.0, "scalar i32 value (got %.6f)", tensor_item(x));
}

Test(mlx_training_dtype_dispatch, scalar_f16) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(4.0, /*requires_grad=*/0,
	                                               /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "scalar dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 4.0, 0.01, "scalar f16 value (got %.6f)", tensor_item(x));
}

/* ---------------------------------------------------------------------
 * tensor_create_streamed (rank-N via shape/rank) — f64 / bf16 / f16
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, rankn_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {2, 3};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 6), shape, /*rank=*/2,
	                                        /*requires_grad=*/0, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "rankN dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 6, "rankN f64 numel should be 6 (got %d)", tensor_numel(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "rankN f64 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, rankn_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	int shape[] = {2, 2};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 4), shape, /*rank=*/2,
	                                        /*requires_grad=*/0, /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "rankN dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "rankN bf16 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, rankn_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	int shape[] = {4};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 4), shape, /*rank=*/1,
	                                        /*requires_grad=*/0, /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "rankN dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "rankN f16 [%d] (got %.9f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_1d_streamed — f16 (f32/f64/i32 covered in core/lifecycle)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, vec1d_f64) {
	param_clear();
	double xd[] = {1.5, -2.5, 3.5};
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(xd, 3), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "1d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(tensor_item_1d(x, i), xd[i], TEST_TOL_RELAXED, "1d f64 [%d] (got %.9f)",
		                   i, tensor_item_1d(x, i));
	}
}

Test(mlx_training_dtype_dispatch, vec1d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(xd, 3), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "1d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(tensor_item_1d(x, i), xd[i], 0.01, "1d f16 [%d] (got %.9f)", i,
		                   tensor_item_1d(x, i));
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_2d_streamed — f64 / bf16 / f16
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, mat2d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "2d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "2d f64 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, mat2d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "2d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "2d bf16 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, mat2d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	TensorHandle x = tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "2d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "2d f16 [%d] (got %.9f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_param_1d_streamed — f64 / bf16 / f16
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, param1d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param1d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "param1d f64 [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param1d_bf16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param1d dtag=17 should be BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.05, "param1d bf16 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param1d_f16) {
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param1d dtag=13 should be F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.01, "param1d f16 [%d] (got %.9f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_param_2d_streamed — f32 / f64 / i32 (all uncovered)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, param2d_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 3, hcopy(xd, 6), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param2d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 6, "param2d f32 numel should be 6 (got %d)", tensor_numel(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "param2d f32 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param2d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param2d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "param2d f64 [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param2d_i32) {
	param_clear();
	double xd[] = {1.0, -2.0, 3.0, -4.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param2d dtag=10 should be I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param2d i32 [%d] (got %.6f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_param_3d_streamed — f32 / f64 / i32 (all uncovered)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, param3d_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param3d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "param3d should be rank-3 (got %d)", tensor_dim(x));
	cr_assert_eq(tensor_numel(x), 8, "param3d f32 numel should be 8 (got %d)", tensor_numel(x));
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "param3d f32 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param3d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param3d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "param3d f64 [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param3d_i32) {
	param_clear();
	double xd[] = {5.0, -6.0, 7.0, -8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param3d dtag=10 should be I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param3d i32 [%d] (got %.6f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_param_4d_streamed — f32 / f64 / i32 (all uncovered)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, param4d_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param4d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "param4d should be rank-4 (got %d)", tensor_dim(x));
	cr_assert_eq(tensor_numel(x), 8, "param4d f32 numel should be 8 (got %d)", tensor_numel(x));
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "param4d f32 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param4d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param4d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "param4d f64 [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, param4d_i32) {
	param_clear();
	double xd[] = {9.0, -10.0, 11.0, -12.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "param4d dtag=10 should be I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "param4d i32 [%d] (got %.6f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_state_1d_streamed — f32 / f64 / i32 (all uncovered)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, state1d_f32) {
	param_clear();
	double xd[] = {1.5, 2.5, 3.5};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "state1d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "state1d f32 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, state1d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "state1d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "state1d f64 [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, state1d_i32) {
	param_clear();
	double xd[] = {13.0, -14.0, 15.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "state1d dtag=10 should be I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "state1d i32 [%d] (got %.6f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_create_state_2d_streamed — f32 / f64 / i32 (all uncovered)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, state2d_f32) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "state2d dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 4, "state2d f32 numel should be 4 (got %d)", tensor_numel(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "state2d f32 [%d] (got %.9f)", i, buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, state2d_f64) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "state2d dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "state2d f64 [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

Test(mlx_training_dtype_dispatch, state2d_i32) {
	param_clear();
	double xd[] = {16.0, -17.0, 18.0, -19.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "state2d dtag=10 should be I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "state2d i32 [%d] (got %.6f)", i, buf[i]);
	}
}

/* ---------------------------------------------------------------------
 * tensor_cast_dtype_streamed — f32 / f64 routes
 * (f16/i32/bf16 covered in core/lifecycle/test_cast_dtype.c; the f32
 *  and f64 case bodies are the remaining uncovered rungs.)
 * ------------------------------------------------------------------- */

Test(mlx_training_dtype_dispatch, cast_to_f64_then_f32) {
	param_clear();
	double xd[] = {1.0, 2.5, -3.75, 0.5};
	TensorHandle x = tensor_create_2d_f64(2, 2, hcopy(xd, 4), 0);
	TensorHandle as_f64 = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(as_f64), "F64", "cast dtag=15 should be F64 (got %s)",
	                 tensor_dtype_name(as_f64));
	TensorHandle as_f32 = tensor_cast_dtype_streamed(as_f64, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(as_f32), "F32", "cast dtag=14 should be F32 (got %s)",
	                 tensor_dtype_name(as_f32));
	double buf[4];
	tensor_to_doubles(as_f32, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED, "cast round-trip [%d] (got %.9f)", i,
		                   buf[i]);
	}
}

#endif /* BACKEND_MLX */

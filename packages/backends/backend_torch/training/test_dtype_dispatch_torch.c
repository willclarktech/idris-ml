/* torch-only Criterion suite for the dtype-dispatch + streamed-shell
 * creator layer.
 *
 * Targets the torch-specific dtag routing in
 * training/dtype_dispatch.cpp (63%) and the shared streamed-shell
 * wrappers in shared/training/dtype_streamed.c (29%) that the common
 * tape suite never reaches:
 *
 *   - tensor_create_*_streamed with the F32 dtag (14) -> the explicit
 *     tensor_create_*_f32 wrappers (cast-after-clone, leaf discipline).
 *   - tensor_create_*_streamed with BF16 (17) / F16 (13) / I32 (10) /
 *     Bool (1) dtags -> the generic create_*_dt / make_param_leaf /
 *     make_state_persistent paths via st_for_dtag.
 *   - tensor_create_param_*_streamed / tensor_create_state_*_streamed
 *     for ranks 1-4 (param) and 1-2 (state).
 *   - tensor_cast_dtype_streamed -> torch_cast_dtype_dtag.
 *
 * Routing goes through the active port (g_active_port.create_*), so a
 * streamed call exercises BOTH the shared shell AND the torch dispatcher
 * in one shot. stream_tag is ignored by torch (single-stream); pass 0.
 *
 * torch CPU base dtype is F64; integer/power-of-two values are exact in
 * every float dtype, so value asserts use whole numbers. F32 roundtrips
 * assert at 1e-5; BF16/F16 at TEST_TOL_RELAXED; integer dtypes exact.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* DType.Core dtag values (kind-major layout): 1=Bool, 10=I32, 13=F16,
   14=F32, 15=F64, 17=BF16. */
#define DTAG_BOOL 1
#define DTAG_I32 10
#define DTAG_F16 13
#define DTAG_F32 14
#define DTAG_F64 15
#define DTAG_BF16 17

/* Streamed creators are callee-owns on the data arg (they free it), so
   every host buffer handed in must be a fresh heap copy. */

/* ---- streamed create (multi-rank, F32 + inference dtags) ---- */

Test(torch_training_dtype_dispatch, streamed_nd_f32) {
	/* tensor_create_streamed dtag=14 -> torch_create_dtag -> tensor_create_f32. */
	double xd[] = {1.5, -2.25, 3.75, 0.5};
	int shape[] = {2, 2};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 4), shape, 2, /*rg=*/0,
	                                        /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "dtag=14 nd should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 4, "numel should be 4");
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 nd readback [%d]: exp %.3f got %.9f", i, xd[i],
		                   buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_nd_bf16) {
	/* tensor_create_streamed dtag=17 -> torch_create_dtag default -> create_nd_dt. */
	double xd[] = {1.0, 2.0, 4.0, 8.0};
	int shape[] = {4};
	TensorHandle x = tensor_create_streamed(hcopy(xd, 4), shape, 1, /*rg=*/0,
	                                        /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "dtag=17 nd should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "BF16 nd readback [%d]: exp %.1f got %.9f", i, xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_scalar_f16) {
	/* tensor_create_scalar_streamed dtag=13 -> torch_create_scalar_dtag default ->
	 * create_scalar_dt. */
	TensorHandle x = tensor_create_scalar_streamed(16.0, /*rg=*/0, /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "dtag=13 scalar should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 16.0, TEST_TOL_RELAXED, "F16 scalar exp 16 got %.6f",
	                   tensor_item(x));
}

Test(torch_training_dtype_dispatch, streamed_scalar_f32) {
	/* tensor_create_scalar_streamed dtag=14 -> tensor_create_scalar_f32 branch. */
	TensorHandle x = tensor_create_scalar_streamed(-3.5, /*rg=*/0, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "dtag=14 scalar should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), -3.5, 1e-5, "F32 scalar exp -3.5 got %.9f", tensor_item(x));
}

Test(torch_training_dtype_dispatch, streamed_1d_f32) {
	/* tensor_create_1d_streamed dtag=14 -> torch_create_1d_dtag -> tensor_create_1d_f32. */
	double xd[] = {2.0, 4.0, 6.0};
	TensorHandle x =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*rg=*/0, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "1d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 1d readback [%d]: exp %.1f got %.9f", i, xd[i],
		                   buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_1d_i32) {
	/* tensor_create_1d_streamed dtag=10 -> torch_create_1d_dtag default -> create_1d_dt. */
	double xd[] = {7.0, -8.0, 100.0};
	TensorHandle x =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*rg=*/0, /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "1d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], 0.0, "I32 1d readback [%d]: exp %.0f got %.6f", i, xd[i],
		                   buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_2d_f32) {
	/* tensor_create_2d_streamed dtag=14 -> torch_create_2d_dtag -> tensor_create_2d_f32. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x =
	    tensor_create_2d_streamed(2, 3, hcopy(xd, 6), /*rg=*/0, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "2d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	cr_assert_eq(tensor_size(x, 1), 3, "dim 1 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 2d readback [%d]: exp %.1f got %.9f", i, xd[i],
		                   buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_2d_bool) {
	/* tensor_create_2d_streamed dtag=1 -> torch_create_2d_dtag default -> create_2d_dt (Bool). */
	double xd[] = {1.0, 0.0, 0.0, 1.0};
	TensorHandle x =
	    tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*rg=*/0, /*stream_tag=*/0, DTAG_BOOL);
	cr_assert_str_eq(tensor_dtype_name(x), "BOOL", "2d dtag=1 should yield BOOL (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], 0.0, "BOOL 2d readback [%d]: exp %.0f got %.6f", i, xd[i],
		                   buf[i]);
}

/* ---- streamed param creators (ranks 1-4, F32 + inference dtags) ---- */

Test(torch_training_dtype_dispatch, streamed_param_1d_f32_grad) {
	/* tensor_create_param_1d_streamed dtag=14 -> tensor_create_param_1d_f32 ->
	   make_param_leaf (cast-before-grad leaf). sum->backward gives grad 1. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param 1d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	param_register("p1", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5,
		                   "grad p1[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
}

Test(torch_training_dtype_dispatch, streamed_param_1d_bf16) {
	/* tensor_create_param_1d_streamed dtag=17 -> make_param_leaf (BF16). BF16 is
	   floating, so the leaf is grad-eligible. */
	double xd[] = {1.0, 2.0};
	TensorHandle x = tensor_create_param_1d_streamed(2, hcopy(xd, 2), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param 1d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[2];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 2; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "BF16 param readback [%d]: exp %.1f got %.9f", i, xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_param_2d_f32) {
	/* tensor_create_param_2d_streamed dtag=14 -> tensor_create_param_2d_f32. */
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param 2d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 param 2d readback [%d]: exp %.1f got %.9f", i,
		                   xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_param_2d_f16) {
	/* tensor_create_param_2d_streamed dtag=13 -> make_param_leaf default (F16). */
	double xd[] = {2.0, 4.0, 8.0, 16.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param 2d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "F16 param 2d readback [%d]: exp %.1f got %.9f", i, xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_param_3d_f32) {
	/* tensor_create_param_3d_streamed dtag=14 -> tensor_create_param_3d_f32. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param 3d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
}

Test(torch_training_dtype_dispatch, streamed_param_3d_bf16) {
	/* tensor_create_param_3d_streamed dtag=17 -> make_param_leaf default (BF16). */
	double xd[] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "param 3d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
}

Test(torch_training_dtype_dispatch, streamed_param_4d_f32) {
	/* tensor_create_param_4d_streamed dtag=14 -> tensor_create_param_4d_f32. */
	double xd[16];
	for (int i = 0; i < 16; i++)
		xd[i] = (double)(i + 1);
	TensorHandle x =
	    tensor_create_param_4d_streamed(2, 2, 2, 2, hcopy(xd, 16), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param 4d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	cr_assert_eq(tensor_numel(x), 16, "numel should be 16");
}

Test(torch_training_dtype_dispatch, streamed_param_4d_f16) {
	/* tensor_create_param_4d_streamed dtag=13 -> make_param_leaf default (F16).
	   make_param_leaf calls requires_grad_(true) unconditionally, so the dtype
	   MUST be floating here — an integer dtag (I32/Bool) would throw c10::Error
	   ("only Tensors of floating point ... can require gradients"). F16 keeps
	   the leaf grad-eligible. */
	double xd[16];
	for (int i = 0; i < 16; i++)
		xd[i] = (double)(i + 1);
	TensorHandle x =
	    tensor_create_param_4d_streamed(2, 2, 2, 2, hcopy(xd, 16), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param 4d dtag=13 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 16, "numel should be 16");
}

/* ---- streamed state creators (ranks 1-2, F32 + inference dtags) ---- */

Test(torch_training_dtype_dispatch, streamed_state_1d_f32) {
	/* tensor_create_state_1d_streamed dtag=14 -> tensor_create_state_1d_f32. */
	double xd[] = {1.5, 2.5, 3.5};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "state 1d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 state 1d readback [%d]: exp %.1f got %.9f", i,
		                   xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_state_1d_bf16) {
	/* tensor_create_state_1d_streamed dtag=17 -> make_state_persistent default (BF16). */
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "state 1d dtag=17 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "BF16 state 1d readback [%d]: exp %.1f got %.9f", i, xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_state_2d_f32) {
	/* tensor_create_state_2d_streamed dtag=14 -> tensor_create_state_2d_f32. */
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "state 2d dtag=14 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "F32 state 2d readback [%d]: exp %.1f got %.9f", i,
		                   xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_state_2d_i32) {
	/* tensor_create_state_2d_streamed dtag=10 -> make_state_persistent default (I32). */
	double xd[] = {10.0, 20.0, 30.0, 40.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32", "state 2d dtag=10 should yield I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], 0.0, "I32 state 2d readback [%d]: exp %.0f got %.6f", i,
		                   xd[i], buf[i]);
}

/* ---- streamed cast ---- */

Test(torch_training_dtype_dispatch, streamed_cast_f64_to_f32) {
	/* tensor_cast_dtype_streamed -> torch_cast_dtype_dtag (F64 -> F32). */
	double xd[] = {1.25, -2.5, 3.0};
	int shape[] = {3};
	TensorHandle src =
	    tensor_create_streamed(hcopy(xd, 3), shape, 1, /*rg=*/0, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(src), "F64", "source should be F64 (got %s)",
	                 tensor_dtype_name(src));
	TensorHandle dst = tensor_cast_dtype_streamed(src, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(dst), "F32", "cast target should be F32 (got %s)",
	                 tensor_dtype_name(dst));
	double buf[3];
	tensor_to_doubles(dst, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-5, "cast F32 readback [%d]: exp %.3f got %.9f", i,
		                   xd[i], buf[i]);
}

Test(torch_training_dtype_dispatch, streamed_cast_f64_to_bf16) {
	/* tensor_cast_dtype_streamed -> torch_cast_dtype_dtag (F64 -> BF16). */
	double xd[] = {2.0, 4.0, 8.0};
	int shape[] = {3};
	TensorHandle src =
	    tensor_create_streamed(hcopy(xd, 3), shape, 1, /*rg=*/0, /*stream_tag=*/0, DTAG_F64);
	TensorHandle dst = tensor_cast_dtype_streamed(src, /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(dst), "BF16", "cast target should be BF16 (got %s)",
	                 tensor_dtype_name(dst));
	double buf[3];
	tensor_to_doubles(dst, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_RELAXED,
		                   "cast BF16 readback [%d]: exp %.1f got %.9f", i, xd[i], buf[i]);
}

/* ---- requires_grad branch on inference-dtype nd creator ---- */

Test(torch_training_dtype_dispatch, streamed_nd_bf16_requires_grad) {
	/* dtag=17 with rg=1: create_nd_dt hits the floating-dtype requires_grad
	   branch (BF16 is floating). Drives the `rg != 0 && is_floating` arm. */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	int shape[] = {3};
	TensorHandle x =
	    tensor_create_streamed(hcopy(xd, 3), shape, 1, /*rg=*/1, /*stream_tag=*/0, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(x), "BF16", "rg-bf16 should yield BF16 (got %s)",
	                 tensor_dtype_name(x));
	param_register("g", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED,
		                   "grad g[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
}

#endif /* BACKEND_TORCH */

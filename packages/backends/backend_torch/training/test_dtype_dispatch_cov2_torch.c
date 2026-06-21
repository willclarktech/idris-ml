/* torch-only Criterion suite #2 for the dtype-dispatch creator layer.
 *
 * Complements test_dtype_dispatch_torch.c (which drives the F32 (dtag
 * 14) explicit-suffix path and the BF16/F16/I32/Bool generic `default`
 * arms). This file closes the remaining FLOAT dtag arms in
 * training/dtype_dispatch.cpp that the first suite never reaches:
 *
 *   - The F64 (dtag 15) `case 15:` arm of every per-shape dtag
 *     dispatcher -> the `tensor_create_*_f64` explicit-suffix wrappers
 *     (tensor_create_1d_f64, tensor_create_2d_f64,
 *     tensor_create_param_{1,2,3,4}d_f64, tensor_create_state_{1,2}d_f64)
 *     and torch_create_scalar_dtag's case 15.
 *   - make_param_leaf's F64 no-cast/no-move fast path (need_cast=false,
 *     need_move=false on a CPU build) — distinct from the F32 cast path
 *     the first suite drives.
 *   - The `rg != 0` requires_grad branch of tensor_create_1d_f32 /
 *     tensor_create_2d_f32 / tensor_create_1d_f64 / tensor_create_2d_f64
 *     (the first suite creates all nd/1d/2d non-param tensors with rg=0).
 *   - create_scalar_dt's `rg && is_floating` requires_grad branch.
 *   - st_for_dtag's case 15 (Float64) via a cast-to-F64.
 *
 * Routing goes through the active port (tensor_create_*_streamed), so a
 * streamed call exercises BOTH the shared shell AND the torch dispatcher
 * in one shot. stream_tag is ignored by torch (single-stream); pass 0.
 *
 * torch CPU base dtype is F64, so F64 roundtrips are exact: value asserts
 * use TEST_TOL_TIGHT. F32 roundtrips of whole/dyadic values assert at
 * 1e-5; F16 at TEST_TOL_RELAXED. NOTE: I32/Bool param-leaf arms are NOT
 * tested — make_param_leaf forces requires_grad_(true) which c10-throws on
 * integer dtypes (the abort guard is excluded from coverage by policy).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* DType.Core dtag values (kind-major layout). */
#define DTAG_F16 13
#define DTAG_F32 14
#define DTAG_F64 15

/* Streamed creators are callee-owns on the data arg (they free it), so
   every host buffer handed in must be a fresh heap copy via hcopy. */

/* ---- F64 (case 15) non-param creators ---- */

Test(dtype_dispatch_cov2, streamed_scalar_f64) {
	/* tensor_create_scalar_streamed dtag=15 -> torch_create_scalar_dtag
	   case 15 -> tensor_create_scalar_f64. */
	TensorHandle x = tensor_create_scalar_streamed(7.25, /*rg=*/0, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "dtag=15 scalar should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 7.25, TEST_TOL_TIGHT, "F64 scalar exp 7.25 got %.15f",
	                   tensor_item(x));
}

Test(dtype_dispatch_cov2, streamed_1d_f64) {
	/* tensor_create_1d_streamed dtag=15 -> torch_create_1d_dtag case 15 ->
	   tensor_create_1d_f64 (rg=0 path). */
	double xd[] = {1.5, -2.25, 3.75};
	TensorHandle x =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*rg=*/0, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "1d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 3, "numel should be 3");
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT,
		                   "F64 1d readback [%d]: exp %.3f got %.15f", i, xd[i], buf[i]);
}

Test(dtype_dispatch_cov2, streamed_2d_f64) {
	/* tensor_create_2d_streamed dtag=15 -> torch_create_2d_dtag case 15 ->
	   tensor_create_2d_f64 (rg=0 path). */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x =
	    tensor_create_2d_streamed(2, 3, hcopy(xd, 6), /*rg=*/0, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "2d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	cr_assert_eq(tensor_size(x, 1), 3, "dim 1 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT,
		                   "F64 2d readback [%d]: exp %.1f got %.15f", i, xd[i], buf[i]);
}

/* ---- F64 (case 15) param creators -> make_param_leaf F64 fast path ---- */

Test(dtype_dispatch_cov2, streamed_param_1d_f64_grad) {
	/* tensor_create_param_1d_streamed dtag=15 -> tensor_create_param_1d_f64 ->
	   make_param_leaf (F64: need_cast=false, need_move=false -> .to() skipped,
	   leaf preserved). sum->backward gives grad 1 per element. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param 1d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	param_register("p1", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT,
		                   "grad p1[%d] should be 1 (got %.15f)", i, param_grad_item_at(0, i));
	param_clear();
}

Test(dtype_dispatch_cov2, streamed_param_2d_f64_grad) {
	/* tensor_create_param_2d_streamed dtag=15 -> tensor_create_param_2d_f64. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_param_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param 2d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	param_register("p2", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT,
		                   "grad p2[%d] should be 1 (got %.15f)", i, param_grad_item_at(0, i));
	param_clear();
}

Test(dtype_dispatch_cov2, streamed_param_3d_f64) {
	/* tensor_create_param_3d_streamed dtag=15 -> tensor_create_param_3d_f64. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param 3d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "F64 param 3d readback [%d]: exp %.1f", i,
		                   xd[i]);
}

Test(dtype_dispatch_cov2, streamed_param_4d_f64) {
	/* tensor_create_param_4d_streamed dtag=15 -> tensor_create_param_4d_f64. */
	double xd[16];
	for (int i = 0; i < 16; i++)
		xd[i] = (double)(i + 1);
	TensorHandle x =
	    tensor_create_param_4d_streamed(2, 2, 2, 2, hcopy(xd, 16), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param 4d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	cr_assert_eq(tensor_numel(x), 16, "numel should be 16");
	double buf[16];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 16; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "F64 param 4d readback [%d]: exp %.1f", i,
		                   xd[i]);
}

/* ---- F64 (case 15) state creators ---- */

Test(dtype_dispatch_cov2, streamed_state_1d_f64) {
	/* tensor_create_state_1d_streamed dtag=15 -> tensor_create_state_1d_f64. */
	double xd[] = {1.5, 2.5, 3.5};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "state 1d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "F64 state 1d readback [%d]: exp %.1f", i,
		                   xd[i]);
}

Test(dtype_dispatch_cov2, streamed_state_2d_f64) {
	/* tensor_create_state_2d_streamed dtag=15 -> tensor_create_state_2d_f64. */
	double xd[] = {10.0, 20.0, 30.0, 40.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "state 2d dtag=15 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], xd[i], TEST_TOL_TIGHT, "F64 state 2d readback [%d]: exp %.1f", i,
		                   xd[i]);
}

/* ---- requires_grad branches of the 1d/2d f32/f64 non-param creators ---- */

Test(dtype_dispatch_cov2, streamed_1d_f64_requires_grad) {
	/* tensor_create_1d_streamed dtag=15 rg=1 -> tensor_create_1d_f64 hits the
	   `if (rg != 0) requires_grad_(true)` branch. sum->backward grad 1. */
	param_clear();
	double xd[] = {3.0, 6.0, 9.0};
	TensorHandle x =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*rg=*/1, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "rg-1d-f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	param_register("g64", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT,
		                   "grad g64[%d] should be 1 (got %.15f)", i, param_grad_item_at(0, i));
	param_clear();
}

Test(dtype_dispatch_cov2, streamed_1d_f32_requires_grad) {
	/* tensor_create_1d_streamed dtag=14 rg=1 -> tensor_create_1d_f32 hits its
	   `if (rg != 0) requires_grad_(true)` branch (cast-before-grad leaf). */
	param_clear();
	double xd[] = {2.0, 4.0, 8.0};
	TensorHandle x =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*rg=*/1, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "rg-1d-f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	param_register("g32", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5,
		                   "grad g32[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	param_clear();
}

Test(dtype_dispatch_cov2, streamed_2d_f64_requires_grad) {
	/* tensor_create_2d_streamed dtag=15 rg=1 -> tensor_create_2d_f64 grad branch. */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*rg=*/1, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "rg-2d-f64 should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	param_register("g64b", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT,
		                   "grad g64b[%d] should be 1 (got %.15f)", i, param_grad_item_at(0, i));
	param_clear();
}

Test(dtype_dispatch_cov2, streamed_2d_f32_requires_grad) {
	/* tensor_create_2d_streamed dtag=14 rg=1 -> tensor_create_2d_f32 grad branch. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0, 8.0};
	TensorHandle x =
	    tensor_create_2d_streamed(2, 2, hcopy(xd, 4), /*rg=*/1, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "rg-2d-f32 should yield F32 (got %s)",
	                 tensor_dtype_name(x));
	param_register("g32b", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5,
		                   "grad g32b[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	param_clear();
}

/* ---- create_scalar_dt requires_grad floating branch ---- */

Test(dtype_dispatch_cov2, streamed_scalar_f16_requires_grad) {
	/* tensor_create_scalar_streamed dtag=13 rg=1 -> torch_create_scalar_dtag
	   default -> create_scalar_dt hits the `rg != 0 && is_floating` branch
	   (F16 is floating). Value asserted; the rg arm executes regardless. */
	TensorHandle x = tensor_create_scalar_streamed(8.0, /*rg=*/1, /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "rg-scalar-f16 should yield F16 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_float_eq(tensor_item(x), 8.0, TEST_TOL_RELAXED, "F16 scalar exp 8 got %.6f",
	                   tensor_item(x));
}

/* ---- st_for_dtag case 15 via cast-to-F64 ---- */

Test(dtype_dispatch_cov2, streamed_cast_f32_to_f64) {
	/* tensor_cast_dtype_streamed -> torch_cast_dtype_dtag -> st_for_dtag(15)
	   (the case 15 / Float64 arm is only reachable via a cast TO F64; the
	   create dispatchers short-circuit case 15 before st_for_dtag). */
	double xd[] = {1.25, -2.5, 3.0};
	TensorHandle src =
	    tensor_create_1d_streamed(3, hcopy(xd, 3), /*rg=*/0, /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(src), "F32", "source should be F32 (got %s)",
	                 tensor_dtype_name(src));
	TensorHandle dst = tensor_cast_dtype_streamed(src, /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(dst), "F64", "cast target should be F64 (got %s)",
	                 tensor_dtype_name(dst));
	double buf[3];
	tensor_to_doubles(dst, buf);
	/* F32->F64 widening is exact for dyadic source values. */
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], xd[i], 1e-6, "cast F64 readback [%d]: exp %.3f got %.15f", i,
		                   xd[i], buf[i]);
}

#endif /* BACKEND_TORCH */

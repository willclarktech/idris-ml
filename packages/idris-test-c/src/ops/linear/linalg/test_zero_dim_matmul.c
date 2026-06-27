/* Criterion suite for zero-dim BLAS guards on tape's matmul-family kernels.
 *
 * Backstory: hf-llama on `BACKEND=tape TAPE_DTYPE=F32` crashed at the first
 * `runGenerate` matmul with `cblas_sgemm: lda=0 K=0`. cblas_sgemm rejects
 * empty contracted-dimension inputs even though the result is well-defined
 * (a properly-shaped zero tensor). The tape kernels call BLAS without
 * checking; this test pins down the per-kernel guard.
 *
 * Each test feeds a zero-width operand into one of the six BLAS-backed
 * kernels (mm, mv, linear, linear_2d, bmm, bmm_3x3) on both F32 and F64
 * and asserts the result has the right shape + is all-zero.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

/* Unified kind-major dtag layout (see
 * `packages/backends/backend_torch/training/dtype_dispatch.cpp:270` and
 * `packages/backends/backend_tape/training/dtype_dispatch.c:11`):
 *   F32 = 14, F64 = 15.
 * The legacy `tensor_create_f32` aborts on tape (no fp32 arena in the
 * legacy creator); the streamed path supports F32 on all three backends
 * via `tensor_create_streamed(..., dtag=14)`. */
#define DTAG_F32 14
#define DTAG_F64 15

static TensorHandle make_zero_shape_f32(int* shape, int rank) {
	return tensor_create_streamed(NULL, shape, rank, 0, 0, DTAG_F32);
}
static TensorHandle make_zero_shape_f64(int* shape, int rank) {
	return tensor_create_streamed(NULL, shape, rank, 0, 0, DTAG_F64);
}

/* ---- helpers ---- */

static void assert_all_zero(TensorHandle t, int expected_numel) {
	cr_assert_eq(tensor_numel(t), expected_numel, "expected numel=%d, got %d", expected_numel,
	             tensor_numel(t));
	if (expected_numel == 0) return;
	double* out = malloc(sizeof(double) * expected_numel);
	tensor_to_doubles(t, out);
	for (int i = 0; i < expected_numel; i++) {
		cr_assert_float_eq(out[i], 0.0, 1e-30, "out[%d] should be 0.0 (got %.6g)", i, out[i]);
	}
	free(out);
}

/* ---- tensor_mm: K=0 ---- */

Test(linear_linalg_zero_dim, mm_f64_zero_K) {
	/* a=[3,0], b=[0,4], r=[3,4] all zero. */
	int sa[] = {3, 0};
	int sb[] = {0, 4};
	TensorHandle a = make_zero_shape_f64(sa, 2);
	TensorHandle b = make_zero_shape_f64(sb, 2);
	TensorHandle r = tensor_mm(a, b);
	cr_assert_eq(tensor_size(r, 0), 3);
	cr_assert_eq(tensor_size(r, 1), 4);
	assert_all_zero(r, 12);
}

Test(linear_linalg_zero_dim, mm_f32_zero_K) {
	int sa[] = {3, 0};
	int sb[] = {0, 4};
	TensorHandle a = make_zero_shape_f32(sa, 2);
	TensorHandle b = make_zero_shape_f32(sb, 2);
	TensorHandle r = tensor_mm(a, b);
	cr_assert_eq(tensor_size(r, 0), 3);
	cr_assert_eq(tensor_size(r, 1), 4);
	assert_all_zero(r, 12);
}

/* ---- tensor_mv: vector-length = 0 ---- */

Test(linear_linalg_zero_dim, mv_f64_zero_inner) {
	/* mat=[3,0], vec=[0], r=[3] all zero. */
	int sm[] = {3, 0};
	int sv[] = {0};
	TensorHandle mat = make_zero_shape_f64(sm, 2);
	TensorHandle vec = make_zero_shape_f64(sv, 1);
	TensorHandle r = tensor_mv(mat, vec);
	cr_assert_eq(tensor_size(r, 0), 3);
	assert_all_zero(r, 3);
}

Test(linear_linalg_zero_dim, mv_f32_zero_inner) {
	int sm[] = {3, 0};
	int sv[] = {0};
	TensorHandle mat = make_zero_shape_f32(sm, 2);
	TensorHandle vec = make_zero_shape_f32(sv, 1);
	TensorHandle r = tensor_mv(mat, vec);
	cr_assert_eq(tensor_size(r, 0), 3);
	assert_all_zero(r, 3);
}

/* ---- tensor_linear: W=[m,0], x=[0], bias=NULL → r=[m] all zero ---- */

Test(linear_linalg_zero_dim, linear_f64_zero_inner) {
	int sW[] = {3, 0};
	int sx[] = {0};
	TensorHandle W = make_zero_shape_f64(sW, 2);
	TensorHandle x = make_zero_shape_f64(sx, 1);
	TensorHandle r = tensor_linear(W, x, NULL);
	cr_assert_eq(tensor_size(r, 0), 3);
	assert_all_zero(r, 3);
}

Test(linear_linalg_zero_dim, linear_f32_zero_inner) {
	int sW[] = {3, 0};
	int sx[] = {0};
	TensorHandle W = make_zero_shape_f32(sW, 2);
	TensorHandle x = make_zero_shape_f32(sx, 1);
	TensorHandle r = tensor_linear(W, x, NULL);
	cr_assert_eq(tensor_size(r, 0), 3);
	assert_all_zero(r, 3);
}

/* ---- tensor_linear_2d: X=[B,0], W=[o,0], no bias → r=[B,o] all zero ---- */

Test(linear_linalg_zero_dim, linear_2d_f64_zero_inner) {
	int sX[] = {2, 0};
	int sW[] = {4, 0};
	TensorHandle X = make_zero_shape_f64(sX, 2);
	TensorHandle W = make_zero_shape_f64(sW, 2);
	TensorHandle r = tensor_linear_2d(W, X, NULL);
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 4);
	assert_all_zero(r, 8);
}

Test(linear_linalg_zero_dim, linear_2d_f32_zero_inner) {
	int sX[] = {2, 0};
	int sW[] = {4, 0};
	TensorHandle X = make_zero_shape_f32(sX, 2);
	TensorHandle W = make_zero_shape_f32(sW, 2);
	TensorHandle r = tensor_linear_2d(W, X, NULL);
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 4);
	assert_all_zero(r, 8);
}

/* ---- tensor_bmm: a=[B,m,0], b=[0,k] → r=[B,m,k] all zero (b shared) ---- */

Test(linear_linalg_zero_dim, bmm_f64_zero_inner) {
	int sa[] = {2, 3, 0};
	int sb[] = {0, 4};
	TensorHandle a = make_zero_shape_f64(sa, 3);
	TensorHandle b = make_zero_shape_f64(sb, 2);
	TensorHandle r = tensor_bmm(a, b);
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 3);
	cr_assert_eq(tensor_size(r, 2), 4);
	assert_all_zero(r, 24);
}

Test(linear_linalg_zero_dim, bmm_f32_zero_inner) {
	int sa[] = {2, 3, 0};
	int sb[] = {0, 4};
	TensorHandle a = make_zero_shape_f32(sa, 3);
	TensorHandle b = make_zero_shape_f32(sb, 2);
	TensorHandle r = tensor_bmm(a, b);
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 3);
	cr_assert_eq(tensor_size(r, 2), 4);
	assert_all_zero(r, 24);
}

/* ---- tensor_bmm_3x3: a=[B,m,0], b=[B,0,k] → r=[B,m,k] all zero ---- */

Test(linear_linalg_zero_dim, bmm_3x3_f64_zero_inner) {
	int sa[] = {2, 3, 0};
	int sb[] = {2, 0, 4};
	TensorHandle a = make_zero_shape_f64(sa, 3);
	TensorHandle b = make_zero_shape_f64(sb, 3);
	TensorHandle r = tensor_bmm_3x3(a, b);
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 3);
	cr_assert_eq(tensor_size(r, 2), 4);
	assert_all_zero(r, 24);
}

Test(linear_linalg_zero_dim, bmm_3x3_f32_zero_inner) {
	int sa[] = {2, 3, 0};
	int sb[] = {2, 0, 4};
	TensorHandle a = make_zero_shape_f32(sa, 3);
	TensorHandle b = make_zero_shape_f32(sb, 3);
	TensorHandle r = tensor_bmm_3x3(a, b);
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 3);
	cr_assert_eq(tensor_size(r, 2), 4);
	assert_all_zero(r, 24);
}

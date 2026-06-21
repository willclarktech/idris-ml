/* mlx-only Criterion suite for core/lifecycle/batch.cpp.
 *
 * Targets:
 *   - tensor_one_hot: 0/1 pattern in the requested storage dtype, for
 *     each dtag branch (default F32, F64=15, BF16=17, F16=13), plus the
 *     out-of-range token guard (tok < 0 or tok >= vocab leaves a zero
 *     row). 0/1 is exact in every dtype, so all asserts are exact.
 *   - tensor_batch / tensor_unbatch round-trip ([..] x N <-> [N, ...]).
 *
 * tensor_one_hot takes ownership of the tokens buffer and free()s it, so
 * each call passes a fresh heap allocation.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* DType.Core dtag values: 13=F16, 14=F32, 15=F64, 17=BF16. */
#define DTAG_F16 13
#define DTAG_F64 15
#define DTAG_BF16 17

static double* heap_copy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

static int* heap_tokens(const int* src, int n) {
	int* buf = (int*)malloc(n * sizeof(int));
	memcpy(buf, src, n * sizeof(int));
	return buf;
}

/* Helper: assert the flattened one-hot pattern for tokens over a vocab. */
static void assert_one_hot(TensorHandle h, const int* tokens, int n_tokens, int vocab) {
	int const total = n_tokens * vocab;
	double* buf = (double*)malloc((size_t)total * sizeof(double));
	tensor_to_doubles(h, buf);
	for (int i = 0; i < n_tokens; i++) {
		for (int v = 0; v < vocab; v++) {
			double expected = (tokens[i] == v) ? 1.0 : 0.0;
			double got = buf[i * vocab + v];
			cr_assert_float_eq(got, expected, 0.0, "one_hot[%d,%d]: tok=%d expected %.0f got %.6f",
			                   i, v, tokens[i], expected, got);
		}
	}
	free(buf);
}

Test(mlx_core_lifecycle_batch, one_hot_default_f32) {
	int tok[] = {0, 2, 1};
	TensorHandle h = tensor_one_hot(heap_tokens(tok, 3), 3, /*vocab=*/3, /*dtag=*/14);
	cr_assert_str_eq(tensor_dtype_name(h), "F32", "dtag=14 one_hot should be F32 (got %s)",
	                 tensor_dtype_name(h));
	cr_assert_eq(tensor_numel(h), 9, "one_hot numel should be n_tokens*vocab = 9");
	assert_one_hot(h, tok, 3, 3);
}

Test(mlx_core_lifecycle_batch, one_hot_f64_dtag) {
	int tok[] = {1, 0};
	TensorHandle h = tensor_one_hot(heap_tokens(tok, 2), 2, /*vocab=*/4, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(h), "F64", "dtag=15 one_hot should be F64 (got %s)",
	                 tensor_dtype_name(h));
	assert_one_hot(h, tok, 2, 4);
}

Test(mlx_core_lifecycle_batch, one_hot_bf16_dtag) {
	int tok[] = {2, 2};
	TensorHandle h = tensor_one_hot(heap_tokens(tok, 2), 2, /*vocab=*/3, DTAG_BF16);
	cr_assert_str_eq(tensor_dtype_name(h), "BF16", "dtag=17 one_hot should be BF16 (got %s)",
	                 tensor_dtype_name(h));
	assert_one_hot(h, tok, 2, 3);
}

Test(mlx_core_lifecycle_batch, one_hot_f16_dtag) {
	int tok[] = {0, 1, 2};
	TensorHandle h = tensor_one_hot(heap_tokens(tok, 3), 3, /*vocab=*/3, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(h), "F16", "dtag=13 one_hot should be F16 (got %s)",
	                 tensor_dtype_name(h));
	assert_one_hot(h, tok, 3, 3);
}

Test(mlx_core_lifecycle_batch, one_hot_out_of_range_token_is_zero_row) {
	/* tok=-1 and tok=vocab are out of [0,vocab) -> all-zero rows (the
	   `tok >= 0 && tok < vocab` guard). */
	int tok[] = {-1, 1, 3};
	TensorHandle h = tensor_one_hot(heap_tokens(tok, 3), 3, /*vocab=*/3, /*dtag=*/14);
	double buf[9];
	tensor_to_doubles(h, buf);
	/* row 0 (tok=-1): all zero */
	for (int v = 0; v < 3; v++)
		cr_assert_float_eq(buf[v], 0.0, 0.0, "tok=-1 row col %d should be 0 (got %.6f)", v, buf[v]);
	/* row 1 (tok=1): one-hot at col 1 */
	cr_assert_float_eq(buf[3 + 1], 1.0, 0.0, "tok=1 should set col 1");
	cr_assert_float_eq(buf[3 + 0], 0.0, 0.0, "tok=1 col 0 should be 0");
	/* row 2 (tok=3 == vocab): all zero */
	for (int v = 0; v < 3; v++)
		cr_assert_float_eq(buf[6 + v], 0.0, 0.0,
		                   "tok=3 (==vocab) row col %d should be 0 (got %.6f)", v, buf[6 + v]);
}

Test(mlx_core_lifecycle_batch, batch_unbatch_round_trip) {
	/* Stack 3 [2]-vectors -> [3,2], then unbatch back to 3 [2]-vectors. */
	double a[] = {1.0, 2.0};
	double b[] = {3.0, 4.0};
	double c[] = {5.0, 6.0};
	int shp[] = {2};
	TensorHandle hs[3];
	hs[0] = tensor_create_f64(heap_copy(a, 2), shp, 1, 0);
	hs[1] = tensor_create_f64(heap_copy(b, 2), shp, 1, 0);
	hs[2] = tensor_create_f64(heap_copy(c, 2), shp, 1, 0);
	TensorHandle batched = tensor_batch(hs, 3);
	cr_assert_eq(tensor_dim(batched), 2, "batched rank should be 2");
	cr_assert_eq(tensor_size(batched, 0), 3, "batched leading dim should be 3");
	cr_assert_eq(tensor_size(batched, 1), 2, "batched inner dim should be 2");

	int out_count = 0;
	TensorHandle* slices = tensor_unbatch(batched, &out_count);
	cr_assert_eq(out_count, 3, "unbatch should yield 3 slices");
	double expected[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
	for (int i = 0; i < 3; i++) {
		cr_assert_eq(tensor_dim(slices[i]), 1, "slice %d rank should be 1", i);
		double buf[2];
		tensor_to_doubles(slices[i], buf);
		for (int j = 0; j < 2; j++)
			cr_assert_float_eq(buf[j], expected[i][j], 1e-5,
			                   "slice[%d][%d]: expected %.1f got %.9f", i, j, expected[i][j],
			                   buf[j]);
	}
	free(slices);
}

#endif /* BACKEND_MLX */

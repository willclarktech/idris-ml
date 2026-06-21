/* mlx-only Criterion suite for tensor accessors (accessors.cpp).
 *
 * Covers the host-readback bridges the common tape suite doesn't reach
 * on mlx:
 *
 *   - tensor_to_int64: byte-level I64 readout. mlx has no native int64
 *     storage; integer round-trip goes through double (2^53 ceiling),
 *     so whole-number F64 inputs read back exactly.
 *   - tensor_to_floats: the F32 fast-path is hit by the common suite,
 *     but the BF16 / F16 / F64-else widening branches are mlx-only and
 *     reached here by casting first.
 *   - tensor_numel / tensor_dim / tensor_size: shape introspection.
 *
 * mlx default storage is F32; value asserts use F32-grade tolerance.
 * BF16/F16 have far coarser mantissas (8/11 bits) so their readback
 * asserts are deliberately loose.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

Test(mlx_core_lifecycle_accessors, shape_introspection) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x = tensor_create_2d_f64(2, 3, hcopy(xd, 6), 0);
	cr_assert_eq(tensor_numel(x), 6, "numel of 2x3 should be 6 (got %d)", tensor_numel(x));
	cr_assert_eq(tensor_dim(x), 2, "rank of 2x3 should be 2 (got %d)", tensor_dim(x));
	cr_assert_eq(tensor_size(x, 0), 2, "size(0) should be 2 (got %d)", tensor_size(x, 0));
	cr_assert_eq(tensor_size(x, 1), 3, "size(1) should be 3 (got %d)", tensor_size(x, 1));
}

Test(mlx_core_lifecycle_accessors, to_int64_whole_numbers) {
	/* Whole-number F64 input: mlx routes int64 readback through double,
	   so positive/negative/zero whole values truncate exactly. */
	param_clear();
	double xd[] = {1.0, -2.0, 1000.0, -42.0, 0.0};
	TensorHandle x = tensor_create_1d_f64(5, hcopy(xd, 5), 0);
	int64_t buf[5];
	tensor_to_int64(x, buf);
	int64_t expected[] = {1, -2, 1000, -42, 0};
	for (int i = 0; i < 5; i++) {
		cr_assert_eq(buf[i], expected[i], "to_int64[%d] expected %lld got %lld", i,
		             (long long)expected[i], (long long)buf[i]);
	}
}

Test(mlx_core_lifecycle_accessors, to_floats_f32_fastpath) {
	/* Default mlx storage is F32 -> hits the memcpy-style fast path. */
	param_clear();
	double xd[] = {0.5, -1.25, 3.75, 100.0};
	TensorHandle x = tensor_create_1d_f64(4, hcopy(xd, 4), 0);
	float buf[4];
	tensor_to_floats(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq((double)buf[i], xd[i], 1e-5, "to_floats f32 [%d] expected %.4f got %.9f",
		                   i, xd[i], (double)buf[i]);
	}
}

Test(mlx_core_lifecycle_accessors, to_floats_f64_branch) {
	/* Cast to F64 storage first so tensor_to_floats walks the
	   double-source else branch (lines 88-91). */
	param_clear();
	double xd[] = {0.5, -1.25, 3.75, 100.0};
	TensorHandle x = tensor_create_1d_f64(4, hcopy(xd, 4), 0);
	TensorHandle xf64 = tensor_cast_dtype_f64(x);
	cr_assert_str_eq(tensor_dtype_name(xf64), "F64", "cast target should be F64 (got %s)",
	                 tensor_dtype_name(xf64));
	float buf[4];
	tensor_to_floats(xf64, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq((double)buf[i], xd[i], 1e-5,
		                   "to_floats f64-branch [%d] expected %.4f got %.9f", i, xd[i],
		                   (double)buf[i]);
	}
}

Test(mlx_core_lifecycle_accessors, to_floats_bf16_branch) {
	/* Cast to bfloat16 so tensor_to_floats walks the bf16 widen branch
	   (lines 80-83). bf16 keeps only ~8 mantissa bits; pick values that
	   are bf16-exact (powers of two and small integers) and assert loose. */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, -8.0};
	TensorHandle x = tensor_create_1d_f64(4, hcopy(xd, 4), 0);
	TensorHandle xbf = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, /*dtag=*/17);
	cr_assert_str_eq(tensor_dtype_name(xbf), "BF16", "cast target should be BF16 (got %s)",
	                 tensor_dtype_name(xbf));
	float buf[4];
	tensor_to_floats(xbf, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq((double)buf[i], xd[i], 0.05,
		                   "to_floats bf16-branch [%d] expected %.4f got %.9f", i, xd[i],
		                   (double)buf[i]);
	}
}

Test(mlx_core_lifecycle_accessors, to_floats_f16_branch) {
	/* Cast to float16 so tensor_to_floats walks the f16 widen branch
	   (lines 84-87). f16 has 11 mantissa bits; small exact values. */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0, -8.0};
	TensorHandle x = tensor_create_1d_f64(4, hcopy(xd, 4), 0);
	TensorHandle xf16 = tensor_cast_dtype_streamed(x, /*stream_tag=*/0, /*dtag=*/13);
	cr_assert_str_eq(tensor_dtype_name(xf16), "F16", "cast target should be F16 (got %s)",
	                 tensor_dtype_name(xf16));
	float buf[4];
	tensor_to_floats(xf16, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq((double)buf[i], xd[i], 0.01,
		                   "to_floats f16-branch [%d] expected %.4f got %.9f", i, xd[i],
		                   (double)buf[i]);
	}
}

#endif /* BACKEND_MLX */

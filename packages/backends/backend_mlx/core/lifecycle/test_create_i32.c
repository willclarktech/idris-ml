/* mlx-only Criterion suite for the I32 storage dispatch path.
 *
 * Verifies that the dtag=10 (I32) dispatchers route to per-dtype
 * base creators that build mx::int32 storage, and that the readback
 * helpers (tensor_item_1d / tensor_to_doubles, both going through
 * mx_read_double / mx_to_doubles in precision.h) interpret the
 * int32 bits correctly rather than reading them as float32.
 *
 * Before this commit landed, every dispatcher fell through to
 * mlx_dtype_unsupported and abort()ed; readback would silently
 * misinterpret int32 bits as float bits (the pre-2026-05-31 BF16
 * bug pattern). Closes the C-side half of TODO Row 46.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* Idris-side dtag values mirroring DType.Core ("8/9/10/11=I8/I16/I32/I64"). */
#define DTAG_I32 10

Test(mlx_core_lifecycle_create_i32, dispatch_1d_roundtrip) {
	/* Whole-number values cover positives, negatives, and zero so the
	   int32 cast (truncation) is exact. */
	double xd[] = {1.0, -2.0, 1000.0, -42.0, 0.0};
	TensorHandle x = tensor_create_1d_streamed(5, hcopy(xd, 5),
	                                           /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32",
	                 "after I32 dispatch, dtype should be I32 (got %s)", tensor_dtype_name(x));
	for (int i = 0; i < 5; i++) {
		double v = tensor_item_1d(x, i);
		cr_assert_float_eq(v, xd[i], 0.0, "I32 roundtrip [%d]: expected %.0f got %.6f", i, xd[i],
		                   v);
	}
}

Test(mlx_core_lifecycle_create_i32, dispatch_2d_roundtrip_via_to_doubles) {
	double xd[] = {7.0, -8.0, 9.0, -10.0, 11.0, -12.0};
	TensorHandle x = tensor_create_2d_streamed(2, 3, hcopy(xd, 6),
	                                           /*requires_grad=*/0,
	                                           /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32",
	                 "after I32 dispatch, dtype should be I32 (got %s)", tensor_dtype_name(x));
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "I32 to_doubles [%d]: expected %.0f got %.6f", i,
		                   xd[i], buf[i]);
	}
}

Test(mlx_core_lifecycle_create_i32, dispatch_param_1d_roundtrip) {
	double xd[] = {3.0, -4.0, 5.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3),
	                                                 /*stream_tag=*/0, DTAG_I32);
	cr_assert_str_eq(tensor_dtype_name(x), "I32",
	                 "after I32 param dispatch, dtype should be I32 (got %s)",
	                 tensor_dtype_name(x));
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 0.0, "I32 param to_doubles [%d]: expected %.0f got %.6f",
		                   i, xd[i], buf[i]);
	}
}

#endif /* BACKEND_MLX */

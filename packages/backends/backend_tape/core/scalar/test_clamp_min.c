/* tensor_clamp_min (tape) — F32 arm + scalar branch.
 * The F64 vector path is exercised elsewhere; this closes the DT_F32 kernel
 * (built via the streamed dtag-14 path) and the numel==1 make_scalar branch. */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

Test(clamp_min, f32_vector) {
	double d[] = {-2.0, 0.0, 3.0};
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(d, 3), /*rg=*/0, /*stream_tag=*/0, 14);
	TensorHandle r = tensor_clamp_min(x, 0.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "got %s", tensor_dtype_name(r));
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_RELAXED, "clamp_min(-2,0) (got %.6f)", out[0]);
	cr_assert_float_eq(out[2], 3.0, TEST_TOL_RELAXED, "clamp_min(3,0) (got %.6f)", out[2]);
}

Test(clamp_min, f64_scalar) {
	double d[] = {-5.0};
	int sh[] = {1};
	TensorHandle x = tensor_create(d, sh, 1, 0);
	TensorHandle r = tensor_clamp_min(x, -1.0); /* numel==1 -> make_scalar branch */
	cr_assert_float_eq(tensor_item(r), -1.0, TEST_TOL_TIGHT, "clamp_min(-5,-1) (got %.9f)",
	                   tensor_item(r));
}

#endif /* BACKEND_TAPE */

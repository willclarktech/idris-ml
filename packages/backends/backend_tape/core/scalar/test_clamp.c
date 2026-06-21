/* Criterion suite for `tensor_clamp` (element-wise two-sided clamp).
 *
 * Forward-only: clamp is inference-only (no tape entry; the BitNet
 * activation-quant path is no-grad by construction). Was at 0% line
 * coverage — the symbol-probe self-matched its own impl file and
 * reported it "covered" while no test ever ran it.
 *
 * RED before this commit: with the expected values asserted below
 * (e.g. clamp([-2,0.5,3], -1, 1) == [-1, 0.5, 1]) the test executes
 * tensor_clamp; perturbing any expected element fails it, confirming
 * the op is genuinely exercised (not a link-only "green").
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"

/* Streamed creators FREE their data argument (callee-owns); hand each one a
   fresh heap copy so callers' stack buffers stay valid. */
static double* hcopy(const double* s, int n) {
	double* b = malloc((size_t)n * sizeof(double));
	memcpy(b, s, (size_t)n * sizeof(double));
	return b;
}

Test(core_scalar_clamp, two_sided) {
	double td[] = {-2.0, -0.5, 0.0, 0.5, 3.0};
	int s[] = {5};
	TensorHandle t = tensor_create(td, s, 1, 0);
	TensorHandle r = tensor_clamp(t, -1.0, 1.0);
	double out[5];
	tensor_to_doubles(r, out);
	double expected[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
	for (int i = 0; i < 5; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_TIGHT, "clamp[%d]: got %.6f expected %.6f",
		                   i, out[i], expected[i]);
}

Test(core_scalar_clamp, all_in_range_is_passthrough) {
	double td[] = {0.1, -0.2, 0.3};
	int s[] = {3};
	TensorHandle t = tensor_create(td, s, 1, 0);
	TensorHandle r = tensor_clamp(t, -1.0, 1.0);
	double out[3];
	tensor_to_doubles(r, out);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], td[i], TEST_TOL_TIGHT, "passthrough[%d]", i);
}

Test(core_scalar_clamp, scalar) {
	TensorHandle t = tensor_create_scalar(5.0, 0);
	cr_assert_float_eq(tensor_item(tensor_clamp(t, -2.0, 2.0)), 2.0, TEST_TOL_TIGHT);
}

/* F32 lane: drives tensor_clamp_f32 (the DT_F32 dispatch branch + the
 * f32 kernel). F32 readback carries ~1e-6 error; assert at 1e-5. */
Test(core_scalar_clamp, two_sided_f32) {
	double td[] = {-2.0, -0.5, 0.0, 0.5, 3.0};
	TensorHandle t = tensor_create_1d_streamed(5, hcopy(td, 5), 0, 0, 14);
	cr_assert_str_eq(tensor_dtype_name(t), "F32");
	TensorHandle r = tensor_clamp(t, -1.0, 1.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	double out[5];
	tensor_to_doubles(r, out);
	double expected[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
	for (int i = 0; i < 5; i++)
		cr_assert_float_eq(out[i], expected[i], 1e-5, "clamp_f32[%d]: got %.6f expected %.6f", i,
		                   out[i], expected[i]);
}

Test(core_scalar_clamp, scalar_f32) {
	TensorHandle t = tensor_create_scalar_streamed(5.0, 0, 0, 14);
	TensorHandle r = tensor_clamp(t, -2.0, 2.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	cr_assert_float_eq(tensor_item(r), 2.0, 1e-5);
}

/* Criterion suite for `tensor_round` (element-wise round-to-nearest-even).
 *
 * Forward-only: inference path (rint = banker's rounding, matching
 * torch.round / mx::round). Was at 0% line coverage.
 *
 * RED before this commit: the half-to-even cases below (2.5 -> 2, not 3)
 * fail if tensor_round is unrun or uses round-half-away; perturbing any
 * expected value fails the test, confirming the op is exercised.
 */

#include <criterion/criterion.h>
#include "test_helpers.h"

Test(core_scalar_round, half_to_even) {
	double td[] = {0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5};
	int s[] = {7};
	TensorHandle t = tensor_create(td, s, 1, 0);
	TensorHandle r = tensor_round(t);
	double out[7];
	tensor_to_doubles(r, out);
	/* round-half-to-even: ties go to the nearest even integer. */
	double expected[] = {0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0};
	for (int i = 0; i < 7; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_TIGHT,
		                   "round[%d]: got %.6f expected %.6f", i, out[i], expected[i]);
}

Test(core_scalar_round, non_ties) {
	double td[] = {2.4, 2.6, -2.4, -2.6};
	int s[] = {4};
	TensorHandle t = tensor_create(td, s, 1, 0);
	TensorHandle r = tensor_round(t);
	double out[4];
	tensor_to_doubles(r, out);
	double expected[] = {2.0, 3.0, -2.0, -3.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_TIGHT, "round[%d]", i);
}

Test(core_scalar_round, scalar) {
	cr_assert_float_eq(tensor_item(tensor_round(tensor_create_scalar(1.5, 0))), 2.0, TEST_TOL_TIGHT);
}

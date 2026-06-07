/* Criterion suite for tensor_expand_mask (forward).
 *
 *   expand_mask(m, B): [m, n] -> [B, m, n] by replicating along a new
 *   leading axis.
 *
 * Closes the "tensor_expand_mask 0 hits" probe gap on all three backends.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

static double* heap_copy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

Test(linear_shape_expand_mask, replicates_correctly) {
	/* m = [[1, 0]] shape [1, 2], B = 3 -> [[[1, 0]], [[1, 0]], [[1, 0]]]
	 * Flat output: 1, 0, 1, 0, 1, 0. */
	double md[] = {1.0, 0.0};
	TensorHandle mask = tensor_create_2d_f64(1, 2, heap_copy(md, 2), 0);
	TensorHandle r = tensor_expand_mask(mask, 3);
	double buf[6];
	tensor_to_doubles(r, buf);
	double expected[] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], expected[i], TEST_TOL_RELAXED,
		                   "expand_mask[%d] should be %.1f (got %.9f)", i, expected[i], buf[i]);
	}
}

Test(linear_shape_expand_mask, b_equals_one_is_identity) {
	/* Edge case: B=1 just adds a leading dim without changing values. */
	double md[] = {5.0, 7.0, 9.0, 11.0};
	TensorHandle mask = tensor_create_2d_f64(2, 2, heap_copy(md, 4), 0);
	TensorHandle r = tensor_expand_mask(mask, 1);
	double buf[4];
	tensor_to_doubles(r, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], md[i], TEST_TOL_RELAXED,
		                   "expand_mask(B=1)[%d] should be %.1f (got %.9f)", i, md[i], buf[i]);
	}
}

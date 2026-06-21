/* Criterion suite for tensor_expand_mask — colocated with the source
 * (nn/mask/expand_mask.c). Complements linear/shape/test_expand_mask.c
 * (which checks values) by asserting the output rank/shape of the
 * [m,n] -> [B,m,n] broadcast and a larger replication factor.
 *
 * The F64 path of expand_mask.c is fully exercised here + by the
 * linear/shape suite; the remaining uncovered lines (the DT_F32 branch)
 * are unreachable from F64 tape tests by design.
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

Test(nn_mask_expand_mask, output_shape_is_B_m_n) {
	/* [2,3] mask expanded to B=4 -> rank-3 [4,2,3], numel 24. */
	double md[] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
	TensorHandle mask = tensor_create_2d_f64(2, 3, heap_copy(md, 6), 0);
	TensorHandle r = tensor_expand_mask(mask, 4);
	cr_assert_eq(tensor_dim(r), 3, "expand_mask result should be rank-3 (got %d)", tensor_dim(r));
	cr_assert_eq(tensor_size(r, 0), 4, "axis 0 should be B=4 (got %d)", tensor_size(r, 0));
	cr_assert_eq(tensor_size(r, 1), 2, "axis 1 should be m=2 (got %d)", tensor_size(r, 1));
	cr_assert_eq(tensor_size(r, 2), 3, "axis 2 should be n=3 (got %d)", tensor_size(r, 2));
	cr_assert_eq(tensor_numel(r), 24, "numel should be 4*2*3=24 (got %d)", tensor_numel(r));
}

Test(nn_mask_expand_mask, larger_batch_replicates_each_slice) {
	/* [1,2] mask -> B=5 yields 5 identical [1,2] slices flattened. */
	double md[] = {3.0, -2.0};
	TensorHandle mask = tensor_create_2d_f64(1, 2, heap_copy(md, 2), 0);
	TensorHandle r = tensor_expand_mask(mask, 5);
	double buf[10];
	tensor_to_doubles(r, buf);
	for (int bi = 0; bi < 5; bi++) {
		cr_assert_float_eq(buf[bi * 2 + 0], 3.0, TEST_TOL_RELAXED,
		                   "slice %d elt 0 should be 3.0 (got %.9f)", bi, buf[bi * 2 + 0]);
		cr_assert_float_eq(buf[bi * 2 + 1], -2.0, TEST_TOL_RELAXED,
		                   "slice %d elt 1 should be -2.0 (got %.9f)", bi, buf[bi * 2 + 1]);
	}
}

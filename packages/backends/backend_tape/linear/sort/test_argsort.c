/* Criterion suite for tape `tensor_argsort` — forward only (non-differentiable;
   the result tensor has requires_grad=0). Covers the ascending and descending
   F64 comparators. The F32 path (argsort.c F32 comparators + arena output) is
   only reached for DT_F32 tensors and is not exercised by these F64 tests. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_sort_argsort, ascending) {
	/* values [3, 1, 2] -> ascending index order [1, 2, 0]. */
	double d[] = {3.0, 1.0, 2.0};
	int s[] = {3};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_argsort(t, 0, 0);
	cr_assert_eq(tensor_dim(r), 1);
	cr_assert_eq(tensor_size(r, 0), 3);
	cr_assert_eq(tensor_numel(r), 3);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 1.0, 1e-12, "asc[0] should be index 1 (got %.1f)", out[0]);
	cr_assert_float_eq(out[1], 2.0, 1e-12, "asc[1] should be index 2 (got %.1f)", out[1]);
	cr_assert_float_eq(out[2], 0.0, 1e-12, "asc[2] should be index 0 (got %.1f)", out[2]);
}

Test(linear_sort_argsort, descending) {
	/* values [3, 1, 2] -> descending index order [0, 2, 1]. */
	double d[] = {3.0, 1.0, 2.0};
	int s[] = {3};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_argsort(t, 0, 1);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 0.0, 1e-12, "desc[0] should be index 0 (got %.1f)", out[0]);
	cr_assert_float_eq(out[1], 2.0, 1e-12, "desc[1] should be index 2 (got %.1f)", out[1]);
	cr_assert_float_eq(out[2], 1.0, 1e-12, "desc[2] should be index 1 (got %.1f)", out[2]);
}

Test(linear_sort_argsort, already_sorted_ascending) {
	/* Already-ascending input is a stable identity ordering. */
	double d[] = {-2.0, 0.0, 5.0, 9.0};
	int s[] = {4};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_argsort(t, 0, 0);
	double out[4];
	tensor_to_doubles(r, out);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], (double)i, 1e-12, "asc identity[%d] (got %.1f)", i, out[i]);
}

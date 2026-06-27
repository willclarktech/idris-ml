/* Criterion suites for tape `tensor_cat2` — forward/backward base behavior
 * plus rank-preserving + weighted-backward coverage arms. */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(linear_concat_cat2, forward_two_vectors) {
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {4.0, 5.0};
	int sa[] = {3};
	int sb[] = {2};
	TensorHandle a = tensor_create(ad, sa, 1, 0);
	TensorHandle b = tensor_create(bd, sb, 1, 0);
	TensorHandle c = tensor_cat2(a, b);
	cr_assert_eq(tensor_numel(c), 5);
	double out[5];
	tensor_to_doubles(c, out);
	cr_assert_float_eq(out[0], 1.0, 1e-12);
	cr_assert_float_eq(out[1], 2.0, 1e-12);
	cr_assert_float_eq(out[2], 3.0, 1e-12);
	cr_assert_float_eq(out[3], 4.0, 1e-12);
	cr_assert_float_eq(out[4], 5.0, 1e-12);
}

Test(linear_concat_cat2, forward_two_2d_rows_preserves_rank) {
	/* RmsNorm row-fold case: each `processRow` emits a [1, hidden] row;
	 * `foldRows` cat2s them into a [seq, hidden]. The previous tape impl
	 * silently collapsed the result to rank-1 [seq*hidden], which broke
	 * HfLlama's narrow-axis-1 pattern downstream (#396). */
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {4.0, 5.0, 6.0};
	int sa[] = {1, 3};
	int sb[] = {1, 3};
	TensorHandle a = tensor_create(ad, sa, 2, 0);
	TensorHandle b = tensor_create(bd, sb, 2, 0);
	TensorHandle c = tensor_cat2(a, b);
	cr_assert_eq(tensor_dim(c), 2, "cat2 of rank-2 inputs must return rank-2 (got rank=%d)",
	             tensor_dim(c));
	cr_assert_eq(tensor_size(c, 0), 2, "cat2 axis-0 size should be 2 (got %d)", tensor_size(c, 0));
	cr_assert_eq(tensor_size(c, 1), 3, "cat2 axis-1 size should be 3 (got %d)", tensor_size(c, 1));
	double out[6];
	tensor_to_doubles(c, out);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[i], ad[i], 1e-12);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(out[3 + i], bd[i], 1e-12);
}

Test(linear_concat_cat2, backward_splits_grad) {
	/* c = cat2(a, b); loss = sum(c). d_a[i] = 1, d_b[j] = 1. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {4.0, 5.0};
	int sa[] = {3};
	int sb[] = {2};
	TensorHandle a = tensor_create(ad, sa, 1, 1);
	TensorHandle b = tensor_create(bd, sb, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_cat2(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "a's grad[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	for (int j = 0; j < 2; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), 1.0, 1e-12,
		                   "b's grad[%d] should be 1 (got %.6f)", j, param_grad_item_at(1, j));
}

Test(cat2_cov, backward_2d_split_weighted) {
	/* a = [[1,2,3],[4,5,6]] (2x3), b = [[7,8,9]] (1x3); c = cat2(a,b) is
	 * 3x3, split (= a->numel) = 6. loss = sum(c*c) => d_a = 2a, d_b = 2b.
	 * Uniform-grad sum() can't distinguish the split point; the weighted
	 * loss does. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double bd[] = {7.0, 8.0, 9.0};
	TensorHandle a = mk2d(2, 3, ad, 1);
	TensorHandle b = mk2d(1, 3, bd, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_cat2(a, b);
	cr_assert_eq(tensor_dim(c), 2, "cat2 of rank-2 inputs must return rank-2 (got %d)",
	             tensor_dim(c));
	cr_assert_eq(tensor_size(c, 0), 3, "cat2 axis-0 size should be 3 (got %d)", tensor_size(c, 0));
	cr_assert_eq(tensor_size(c, 1), 3, "cat2 axis-1 size should be 3 (got %d)", tensor_size(c, 1));
	TensorHandle sq = tensor_mul(c, c);
	TensorHandle loss = tensor_sum(sq);
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 2.0 * ad[i], TEST_TOL_TIGHT,
		                   "a grad[%d] should be %.6f (got %.6f)", i, 2.0 * ad[i],
		                   param_grad_item_at(0, i));
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), 2.0 * bd[j], TEST_TOL_TIGHT,
		                   "b grad[%d] should be %.6f (got %.6f)", j, 2.0 * bd[j],
		                   param_grad_item_at(1, j));
	param_clear();
}

Test(cat2_cov, backward_only_b_requires_grad) {
	/* a is frozen (requires_grad=0), b is a param. rg = (0 || 1) drives the
	 * second operand of the || and still fires OP_CAT tape_append; backward
	 * scatters grad to b's flat slice [split, total). loss = sum(c). */
	param_clear();
	double ad[] = {1.0, 2.0};
	double bd[] = {3.0, 4.0, 5.0};
	int sa[] = {2};
	int sb[] = {3};
	TensorHandle a = tensor_create(ad, sa, 1, 0);
	TensorHandle b = tensor_create(bd, sb, 1, 1);
	param_register("b", b);
	TensorHandle c = tensor_cat2(a, b);
	cr_assert_eq(tensor_numel(c), 5);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(0, j), 1.0, TEST_TOL_TIGHT,
		                   "b grad[%d] should be 1 (got %.6f)", j, param_grad_item_at(0, j));
	param_clear();
}

/* Coverage suite for tape `tensor_cat2` — exercises the rank-preserving
 * forward + backward arms the base test_cat2.c suite leaves cold:
 *
 *   - the rank-2 forward path WITH requires_grad set (trailing-dim copy
 *     loop + OP_CAT tape_append on the F64 branch),
 *   - the backward scatter with a multi-row split point and a
 *     non-uniform upstream gradient (loss = sum(c*c) => d_c = 2c), which
 *     pins both the `a` arm (flat indices [0, split)) and the `b` arm
 *     (flat indices [split, total)) of tape_backward_cat,
 *   - the `rg = a->requires_grad || b->requires_grad` short-circuit
 *     second-operand-true case (a frozen, b a param).
 *
 * The F32 branch (cat2.c `a->dtype_tag == DT_F32`) is intentionally NOT
 * covered: tape has no fp32 arena, so tensor_create_f32 aborts — that arm
 * is unreachable from a tape build.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

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

/* Criterion suite for `tensor_gather_rows` (row-wise gather).
 * Covers all backends via the public FFI (see mk/tests.mk discovery
 * comment); forward, backward-scatter, and the F32 paired case per
 * docs/develop/coverage-policy.md. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_index_gather_rows, forward_picks_per_row) {
	/* input [3,2] = [[1,2],[3,4],[5,6]], index [1,0,1] -> [2,3,6] */
	double id[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double ixd[] = {1.0, 0.0, 1.0};
	int s_in[] = {3, 2};
	int s_ix[] = {3};
	TensorHandle input = tensor_create(id, s_in, 2, 0);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	TensorHandle r = tensor_gather_rows(input, index, 3, 2);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 2.0, 1e-12, "row 0 picks col 1 (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], 3.0, 1e-12, "row 1 picks col 0 (got %.6f)", out[1]);
	cr_assert_float_eq(out[2], 6.0, 1e-12, "row 2 picks col 1 (got %.6f)", out[2]);
}

Test(linear_index_gather_rows, backward_scatters_to_selected_cells) {
	param_clear();
	double id[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double ixd[] = {1.0, 0.0, 1.0};
	int s_in[] = {3, 2};
	int s_ix[] = {3};
	TensorHandle input = tensor_create(id, s_in, 2, 1);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	param_register("gr_in", input);
	TensorHandle r = tensor_gather_rows(input, index, 3, 2);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	/* d_input = [[0,1],[1,0],[0,1]] flattened; unselected cells 0 */
	double expect[] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expect[i], 1e-12,
		                   "d_input[%d] expect %.1f got %.6f", i, expect[i],
		                   param_grad_item_at(0, i));
}

Test(linear_index_gather_rows, f32_paired_forward_and_grad) {
	/* F32 paired oracle: same computation at F32; forward within 1e-6
	 * of the F64 values, backward exact (small integers). dtag 14 = F32. */
	param_clear();
	double id[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double ixd[] = {1.0, 0.0, 1.0};
	int s_in[] = {3, 2};
	int s_ix[] = {3};
	TensorHandle in32 = tensor_create_streamed(id, s_in, 2, 1, 0, 14);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	param_register("gr_in32", in32);
	TensorHandle r = tensor_gather_rows(in32, index, 3, 2);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 2.0, 1e-6);
	cr_assert_float_eq(out[1], 3.0, 1e-6);
	cr_assert_float_eq(out[2], 6.0, 1e-6);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expect[] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expect[i], 1e-6,
		                   "f32 d_input[%d] expect %.1f got %.6f", i, expect[i],
		                   param_grad_item_at(0, i));
}

/* Criterion suite for `tensor_max_rows` (row-wise max).
 * Covers all backends via the public FFI; forward, backward-to-argmax,
 * and the F32 paired case. Inputs avoid ties (tie-breaking across
 * backends is documented unspecified). */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_index_max_rows, forward_max_per_row) {
	/* input [3,2] = [[1,5],[7,2],[3,4]] -> [5,7,4] */
	double id[] = {1.0, 5.0, 7.0, 2.0, 3.0, 4.0};
	int s_in[] = {3, 2};
	TensorHandle input = tensor_create(id, s_in, 2, 0);
	TensorHandle r = tensor_max_rows(input, 3, 2);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 5.0, 1e-12, "row 0 max (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], 7.0, 1e-12, "row 1 max (got %.6f)", out[1]);
	cr_assert_float_eq(out[2], 4.0, 1e-12, "row 2 max (got %.6f)", out[2]);
}

Test(linear_index_max_rows, backward_routes_to_argmax) {
	param_clear();
	double id[] = {1.0, 5.0, 7.0, 2.0, 3.0, 4.0};
	int s_in[] = {3, 2};
	TensorHandle input = tensor_create(id, s_in, 2, 1);
	param_register("mr_in", input);
	TensorHandle r = tensor_max_rows(input, 3, 2);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	/* argmax cells: (0,1), (1,0), (2,1) -> d_input [[0,1],[1,0],[0,1]] */
	double expect[] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expect[i], 1e-12,
		                   "d_input[%d] expect %.1f got %.6f", i, expect[i],
		                   param_grad_item_at(0, i));
}

Test(linear_index_max_rows, f32_paired_forward_and_grad) {
	param_clear();
	double id[] = {1.0, 5.0, 7.0, 2.0, 3.0, 4.0};
	int s_in[] = {3, 2};
	TensorHandle in32 = tensor_create_streamed(id, s_in, 2, 1, 0, 14);
	param_register("mr_in32", in32);
	TensorHandle r = tensor_max_rows(in32, 3, 2);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 5.0, 1e-6);
	cr_assert_float_eq(out[1], 7.0, 1e-6);
	cr_assert_float_eq(out[2], 4.0, 1e-6);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	double expect[] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expect[i], 1e-6,
		                   "f32 d_input[%d] expect %.1f got %.6f", i, expect[i],
		                   param_grad_item_at(0, i));
}

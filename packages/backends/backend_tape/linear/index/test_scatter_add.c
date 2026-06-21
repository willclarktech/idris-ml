/* Criterion suite for tape `tensor_scatter_add`. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_index_scatter_add, forward_accumulates) {
	/* index = [0, 1, 0, 2], src = [10, 20, 30, 40], out_size = 3
	   r[0] = 10 + 30 = 40, r[1] = 20, r[2] = 40 */
	double ixd[] = {0.0, 1.0, 0.0, 2.0};
	double sd[] = {10.0, 20.0, 30.0, 40.0};
	int s_ix[] = {4};
	int s_s[] = {4};
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	TensorHandle src = tensor_create(sd, s_s, 1, 0);
	TensorHandle r = tensor_scatter_add(index, src, 3);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 40.0, 1e-12);
	cr_assert_float_eq(out[1], 20.0, 1e-12);
	cr_assert_float_eq(out[2], 40.0, 1e-12);
}

Test(linear_index_scatter_add, forward_skips_out_of_range_index) {
	/* index = [-1, 1, 5, 2], out_size = 3. Indices -1 and 5 are out of range
	   and must be dropped by the `idx >= 0 && idx < out_size` guard.
	   r[0] = 0, r[1] = 20, r[2] = 40. */
	double ixd[] = {-1.0, 1.0, 5.0, 2.0};
	double sd[] = {10.0, 20.0, 30.0, 40.0};
	int s_ix[] = {4};
	int s_s[] = {4};
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	TensorHandle src = tensor_create(sd, s_s, 1, 0);
	TensorHandle r = tensor_scatter_add(index, src, 3);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 0.0, 1e-12, "out[0] should be 0 (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], 20.0, 1e-12);
	cr_assert_float_eq(out[2], 40.0, 1e-12);
}

Test(linear_index_scatter_add, backward_skips_out_of_range_index) {
	/* index = [-1, 1, 5, 2]. d_src[i] = d_r[index[i]] when in range, else 0.
	   loss = sum(r) -> d_r = 1 everywhere in range, so
	   d_src = [0, 1, 0, 1] (positions 0 and 2 had out-of-range indices). */
	param_clear();
	double ixd[] = {-1.0, 1.0, 5.0, 2.0};
	double sd[] = {10.0, 20.0, 30.0, 40.0};
	int s_ix[] = {4};
	int s_s[] = {4};
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	TensorHandle src = tensor_create(sd, s_s, 1, 1);
	param_register("src", src);
	TensorHandle r = tensor_scatter_add(index, src, 3);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "src grad[0] should be 0 (got %.6f)",
	                   param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12, "src grad[2] should be 0 (got %.6f)",
	                   param_grad_item_at(0, 2));
	cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12);
}

Test(linear_index_scatter_add, backward_gathers_grad) {
	/* Same setup. loss = sum(r) -> d_src[i] = d_r[index[i]] = 1 for any valid idx. */
	param_clear();
	double ixd[] = {0.0, 1.0, 0.0, 2.0};
	double sd[] = {10.0, 20.0, 30.0, 40.0};
	int s_ix[] = {4};
	int s_s[] = {4};
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	TensorHandle src = tensor_create(sd, s_s, 1, 1);
	param_register("src", src);
	TensorHandle r = tensor_scatter_add(index, src, 3);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "src grad[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
}

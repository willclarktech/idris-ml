/* Criterion suite for tape `tensor_select`. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_shape_select, forward_vector_element) {
	double d[] = {10.0, 20.0, 30.0, 40.0};
	int s[] = {4};
	TensorHandle v = tensor_create(d, s, 1, 0);
	TensorHandle e = tensor_select(v, 0, 2);
	cr_assert_float_eq(tensor_item(e), 30.0, 1e-12);
}

Test(linear_shape_select, backward_scatters_to_index) {
	/* Vector [a0, a1, a2, a3]; select index 1; backward should put 1.0
	   at a's grad[1] and 0.0 elsewhere. */
	param_clear();
	double d[] = {10.0, 20.0, 30.0, 40.0};
	int s[] = {4};
	TensorHandle v = tensor_create(d, s, 1, 1);
	param_register("v", v);
	TensorHandle picked = tensor_select(v, 0, 1);
	tensor_backward(picked);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "grad[0] should be 0");
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12, "grad[1] should be 1.0 (got %.6f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12, "grad[2] should be 0");
	cr_assert_float_eq(param_grad_item_at(0, 3), 0.0, 1e-12, "grad[3] should be 0");
}

Test(linear_shape_select, forward_matrix_row) {
	/* [3, 2] matrix; select(dim=0, index=1) -> row 1 as a vector [2, 3]. */
	double d[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	int s[] = {3, 2};
	TensorHandle m = tensor_create(d, s, 2, 0);
	TensorHandle row = tensor_select(m, 0, 1);
	cr_assert_eq(tensor_dim(row), 1, "selected row should be rank-1");
	cr_assert_eq(tensor_size(row, 0), 2);
	double out[2];
	tensor_to_doubles(row, out);
	cr_assert_float_eq(out[0], 2.0, 1e-12);
	cr_assert_float_eq(out[1], 3.0, 1e-12);
}

Test(linear_shape_select, backward_matrix_row_scatters) {
	/* [3, 2]; select row 1; sum the row -> backward should put 1.0 across
	 * row 1 (elements 2,3) and 0.0 elsewhere. Exercises the row-select
	 * backward branch (cols > 1). */
	param_clear();
	double d[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	int s[] = {3, 2};
	TensorHandle m = tensor_create(d, s, 2, 1);
	param_register("m", m);
	TensorHandle loss = tensor_sum(tensor_select(m, 0, 1));
	tensor_backward(loss);
	double expected[6] = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expected[i], 1e-12,
		                   "grad[%d] should be %.1f (got %.6f)", i, expected[i],
		                   param_grad_item_at(0, i));
}

#ifdef BACKEND_TAPE
Test(linear_shape_select, high_rank_fallback_scalar) {
	/* rank-3 select hits the high-rank fallback path: returns a fresh scalar
	 * at the flat `index` via tape_load_d. [2, 2, 2] = 0..7; index 5 -> 5.0. */
	double d[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
	int s[] = {2, 2, 2};
	TensorHandle t = tensor_create(d, s, 3, 0);
	TensorHandle picked = tensor_select(t, 0, 5);
	cr_assert_float_eq(tensor_item(picked), 5.0, 1e-12, "fallback scalar should be 5.0 (got %.6f)",
	                   tensor_item(picked));
}
#endif /* BACKEND_TAPE */

#ifdef BACKEND_TAPE
Test(linear_shape_select, scalar_identity) {
	/* select on a rank-0 scalar is identity (returns the same handle). */
	double d[] = {42.0};
	int s[] = {1};
	TensorHandle scalar = tensor_create(d, s, 0, 0);
	TensorHandle picked = tensor_select(scalar, 0, 0);
	cr_assert_float_eq(tensor_item(picked), 42.0, 1e-12);
}
#endif /* BACKEND_TAPE */

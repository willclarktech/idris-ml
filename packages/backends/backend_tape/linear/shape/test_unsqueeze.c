/* Criterion suite for tape `tensor_unsqueeze` — inserts a size-1 axis at
   `dim` (delegates to tensor_reshape, so backward flows via OP_RESHAPE).
   Includes a death test for the out-of-range guard. */

#include <signal.h>
#include <criterion/criterion.h>
#include "backend.h"

Test(linear_shape_unsqueeze, insert_leading_axis) {
	double d[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle v = tensor_create(d, s, 1, 0);
	TensorHandle u = tensor_unsqueeze(v, 0); /* [3] -> [1,3] */
	cr_assert_eq(tensor_dim(u), 2);
	cr_assert_eq(tensor_size(u, 0), 1);
	cr_assert_eq(tensor_size(u, 1), 3);
	cr_assert_eq(tensor_numel(u), 3);
}

Test(linear_shape_unsqueeze, scalar_to_vector) {
	/* rank-0 scalar -> [1] (the DNC primCat2 path). */
	double d[] = {7.0};
	int s[] = {1};
	TensorHandle scalar = tensor_create(d, s, 0, 0);
	TensorHandle u = tensor_unsqueeze(scalar, 0);
	cr_assert_eq(tensor_dim(u), 1);
	cr_assert_eq(tensor_size(u, 0), 1);
	cr_assert_float_eq(tensor_item(u), 7.0, 1e-12);
}

Test(linear_shape_unsqueeze, backward_passthrough) {
	param_clear();
	double d[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle v = tensor_create(d, s, 1, 1);
	param_register("v", v);
	TensorHandle loss = tensor_sum(tensor_unsqueeze(v, 0));
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "unsqueeze grad[%d] should be 1.0 (got %.6f)", i,
		                   param_grad_item_at(0, i));
}

/* Out-of-range dim aborts (dim > old_rank). The guard's fprintf+abort lines
   are GCOVR_EXCL'd in unsqueeze.c (abort() skips the gcov flush, so the forked
   child can't register them); this death test is what asserts the guard fires. */
Test(linear_shape_unsqueeze, out_of_range_dim_aborts, .signal = SIGABRT) {
	double d[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle v = tensor_create(d, s, 1, 0);
	tensor_unsqueeze(v, 5); /* dim=5 > rank=1 -> abort */
}

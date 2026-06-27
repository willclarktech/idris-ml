/* Criterion suite for tape `tensor_reshape_3d` / `tensor_reshape_4d` —
   fixed-rank reshape delegations. Forward shares storage; backward flows
   through OP_RESHAPE (grad-passthrough). */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_shape_reshape_dims, reshape_3d_forward) {
	double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int s[] = {6};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_reshape_3d(t, 1, 2, 3);
	cr_assert_eq(tensor_dim(r), 3);
	cr_assert_eq(tensor_size(r, 0), 1);
	cr_assert_eq(tensor_size(r, 1), 2);
	cr_assert_eq(tensor_size(r, 2), 3);
	cr_assert_eq(tensor_numel(r), 6);
	double out[6];
	tensor_to_doubles(r, out);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], d[i], 1e-12);
}

Test(linear_shape_reshape_dims, reshape_3d_backward_passthrough) {
	param_clear();
	double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int s[] = {6};
	TensorHandle t = tensor_create(d, s, 1, 1);
	param_register("t", t);
	TensorHandle loss = tensor_sum(tensor_reshape_3d(t, 1, 2, 3));
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "reshape_3d grad[%d] should be 1.0 (got %.6f)", i,
		                   param_grad_item_at(0, i));
}

Test(linear_shape_reshape_dims, reshape_4d_forward) {
	double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int s[] = {6};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_reshape_4d(t, 1, 1, 2, 3);
	cr_assert_eq(tensor_dim(r), 4);
	cr_assert_eq(tensor_size(r, 0), 1);
	cr_assert_eq(tensor_size(r, 1), 1);
	cr_assert_eq(tensor_size(r, 2), 2);
	cr_assert_eq(tensor_size(r, 3), 3);
	cr_assert_eq(tensor_numel(r), 6);
}

Test(linear_shape_reshape_dims, reshape_4d_backward_passthrough) {
	param_clear();
	double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int s[] = {6};
	TensorHandle t = tensor_create(d, s, 1, 1);
	param_register("t", t);
	TensorHandle loss = tensor_sum(tensor_reshape_4d(t, 1, 1, 2, 3));
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "reshape_4d grad[%d] should be 1.0 (got %.6f)", i,
		                   param_grad_item_at(0, i));
}

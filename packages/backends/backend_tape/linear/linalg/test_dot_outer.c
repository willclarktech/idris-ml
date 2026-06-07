/* Criterion suite for tape `tensor_dot` + `tensor_outer`. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_linalg_dot, forward_backward) {
	/* a·b = 1*4 + 2*5 + 3*6 = 32; d/da = b, d/db = a */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {4.0, 5.0, 6.0};
	int s[] = {3};
	TensorHandle a = tensor_create(ad, s, 1, 1);
	TensorHandle b = tensor_create(bd, s, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle r = tensor_dot(a, b);
	cr_assert_float_eq(tensor_item(r), 32.0, 1e-12);
	tensor_backward(r);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), bd[i], 1e-12,
		                   "d_a[%d] should be b[%d]=%.1f (got %.6f)", i, i, bd[i],
		                   param_grad_item_at(0, i));
		cr_assert_float_eq(param_grad_item_at(1, i), ad[i], 1e-12,
		                   "d_b[%d] should be a[%d]=%.1f (got %.6f)", i, i, ad[i],
		                   param_grad_item_at(1, i));
	}
}

Test(linear_linalg_outer, forward_backward) {
	/* outer([1, 2], [3, 4, 5]) = [[3, 4, 5], [6, 8, 10]]
	   loss = sum -> d_a[i] = sum_j b[j]; d_b[j] = sum_i a[i] */
	param_clear();
	double ad[] = {1.0, 2.0};
	double bd[] = {3.0, 4.0, 5.0};
	int sa[] = {2};
	int sb[] = {3};
	TensorHandle a = tensor_create(ad, sa, 1, 1);
	TensorHandle b = tensor_create(bd, sb, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle r = tensor_outer(a, b);
	double out[6];
	tensor_to_doubles(r, out);
	double expected[] = {3, 4, 5, 6, 8, 10};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expected[i], 1e-12);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	/* d_a[i] = sum_j b[j] = 3+4+5 = 12; d_b[j] = sum_i a[i] = 1+2 = 3 */
	cr_assert_float_eq(param_grad_item_at(0, 0), 12.0, 1e-12, "d_a[0] should be 12 (got %.6f)",
	                   param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 12.0, 1e-12);
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), 3.0, 1e-12, "d_b[%d] should be 3 (got %.6f)",
		                   j, param_grad_item_at(1, j));
}

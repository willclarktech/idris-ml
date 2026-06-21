/* Criterion suite for tape `tensor_cumprod`. */

#include <criterion/criterion.h>
#include "backend.h"

Test(linear_sort_cumprod, forward) {
	double d[] = {2.0, 3.0, 4.0};
	int s[] = {3};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_cumprod(t, 0);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 2.0, 1e-12);
	cr_assert_float_eq(out[1], 6.0, 1e-12);
	cr_assert_float_eq(out[2], 24.0, 1e-12);
}

Test(linear_sort_cumprod, backward_simple) {
	/* a = [2, 3, 4]; r = [2, 6, 24]; loss = sum(r) = 32
	   d_a[0] = (d_r[0]*r[0] + d_r[1]*r[1] + d_r[2]*r[2]) / a[0]
	          = (2 + 6 + 24) / 2 = 16
	   d_a[1] = (6 + 24) / 3 = 10
	   d_a[2] = 24 / 4 = 6 */
	param_clear();
	double d[] = {2.0, 3.0, 4.0};
	int s[] = {3};
	TensorHandle t = tensor_create(d, s, 1, 1);
	param_register("t", t);
	TensorHandle r = tensor_cumprod(t, 0);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 16.0, 1e-10, "d_t[0] should be 16 (got %.6f)",
	                   param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 10.0, 1e-10, "d_t[1] should be 10 (got %.6f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 6.0, 1e-10, "d_t[2] should be 6 (got %.6f)",
	                   param_grad_item_at(0, 2));
}

Test(linear_sort_cumprod, backward_with_zero) {
	/* a = [2, 0, 4]; r = [2, 0, 0]; loss = sum(r) = 2.
	   a[1]==0 forces the exclusive-product recompute branch (cumprod.c 59-67).
	   Hand grad (d_r = [1,1,1]):
	     d_a[2]: a[2]=4 != 0 -> suffix_sum so far = d_r[2]*r[2] = 1*0 = 0 -> 0/4 = 0.
	     d_a[1]: a[1]=0 -> exclusive-product over suffix j=1,2:
	             j=1: prod over k<=1 excl k=1 = a[0] = 2; d_r[1]*2 = 2
	             j=2: prod over k<=2 excl k=1 = a[0]*a[2] = 8; d_r[2]*8 = 8
	             partial = 2 + 8 = 10
	     d_a[0]: a[0]=2 != 0 -> suffix_sum = d_r[0]*r[0] + d_r[1]*r[1] + d_r[2]*r[2]
	                                       = 1*2 + 1*0 + 1*0 = 2 -> 2/2 = 1. */
	param_clear();
	double d[] = {2.0, 0.0, 4.0};
	int s[] = {3};
	TensorHandle t = tensor_create(d, s, 1, 1);
	param_register("t", t);
	TensorHandle r = tensor_cumprod(t, 0);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-10, "d_t[0] should be 1 (got %.6f)",
	                   param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 10.0, 1e-10, "d_t[1] should be 10 (got %.6f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-10, "d_t[2] should be 0 (got %.6f)",
	                   param_grad_item_at(0, 2));
}

Test(linear_sort_cumprod, forward_no_grad) {
	/* requires_grad=0 path: no tape append, plain forward. */
	double d[] = {1.5, 2.0, 0.5, 4.0};
	int s[] = {4};
	TensorHandle t = tensor_create(d, s, 1, 0);
	TensorHandle r = tensor_cumprod(t, 0);
	double out[4];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 1.5, 1e-12);
	cr_assert_float_eq(out[1], 3.0, 1e-12);
	cr_assert_float_eq(out[2], 1.5, 1e-12);
	cr_assert_float_eq(out[3], 6.0, 1e-12);
}

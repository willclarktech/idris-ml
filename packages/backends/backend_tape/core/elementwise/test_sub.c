/* Criterion suite for tape `tensor_sub`. */

#include <criterion/criterion.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

/* Heap-copy a stack array — the streamed creators take ownership and free it. */
static double* hcopy(const double* s, int n) {
	double* b = (double*)malloc((size_t)n * sizeof(double));
	memcpy(b, s, (size_t)n * sizeof(double));
	return b;
}

Test(core_elementwise_sub, forward_scalar) {
	TensorHandle a = tensor_create_scalar(10.0, 0);
	TensorHandle b = tensor_create_scalar(3.0, 0);
	TensorHandle c = tensor_sub(a, b);
	cr_assert_float_eq(tensor_item(c), 7.0, 1e-12);
}

Test(core_elementwise_sub, forward_vector) {
	double ad[] = {10.0, 20.0, 30.0};
	double bd[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle a = tensor_create(ad, s, 1, 0);
	TensorHandle b = tensor_create(bd, s, 1, 0);
	TensorHandle c = tensor_sub(a, b);
	double out[3];
	tensor_to_doubles(c, out);
	cr_assert_float_eq(out[0], 9.0, 1e-12);
	cr_assert_float_eq(out[1], 18.0, 1e-12);
	cr_assert_float_eq(out[2], 27.0, 1e-12);
}

Test(core_elementwise_sub, backward_scalar_grads_signs) {
	/* c = a - b; dc/da = +1, dc/db = -1 */
	param_clear();
	TensorHandle a = tensor_create_scalar(10.0, 1);
	TensorHandle b = tensor_create_scalar(3.0, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	tensor_backward(c);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12, "d(a-b)/da should be +1.0 (got %.6f)",
	                   param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(1, 0), -1.0, 1e-12, "d(a-b)/db should be -1.0 (got %.6f)",
	                   param_grad_item_at(1, 0));
}

Test(core_elementwise_sub, backward_vector_signs) {
	param_clear();
	double ad[] = {5.0, 6.0, 7.0};
	double bd[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle a = tensor_create(ad, s, 1, 1);
	TensorHandle b = tensor_create(bd, s, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "d(sum(a-b))/da[%d] should be +1.0", i);
		cr_assert_float_eq(param_grad_item_at(1, i), -1.0, 1e-12,
		                   "d(sum(a-b))/db[%d] should be -1.0", i);
	}
}

Test(core_elementwise_sub, backward_scalar_minus_vector_a_side) {
	/* c = scalar - vector; exercises the a->numel==1 (scalar-on-a) branch.
	   loss = sum(c) => d_scalar = numel(vec) = 4, d_vec[i] = -1. */
	param_clear();
	TensorHandle a = tensor_create_scalar(5.0, 1);
	double bd[] = {1.0, 2.0, 3.0, 4.0};
	int s[] = {4};
	TensorHandle b = tensor_create(bd, s, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, 1e-12,
	                   "d(sum(scalar-vec))/d_scalar should be +4.0 (got %.6f)",
	                   param_grad_item_at(0, 0));
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), -1.0, 1e-12,
		                   "d(sum(scalar-vec))/d_vec[%d] should be -1.0", i);
}

Test(core_elementwise_sub, backward_vector_minus_scalar_b_side) {
	/* c = vector - scalar; exercises the b->numel==1 (scalar-on-b) branch.
	   loss = sum(c) => d_vec[i] = +1, d_scalar = -numel(vec) = -4. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0};
	int s[] = {4};
	TensorHandle a = tensor_create(ad, s, 1, 1);
	TensorHandle b = tensor_create_scalar(5.0, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "d(sum(vec-scalar))/d_vec[%d] should be +1.0", i);
	cr_assert_float_eq(param_grad_item_at(1, 0), -4.0, 1e-12,
	                   "d(sum(vec-scalar))/d_scalar should be -4.0 (got %.6f)",
	                   param_grad_item_at(1, 0));
}

/* ---- F32 coverage (sub.c fn_sub_f32 + binop_elementwise_f32_disp) ---- */

Test(core_elementwise_sub, f32_forward_scalar) {
	/* Drives sub.c line 26 (F32 dispatch) + F32 both-scalar path. */
	double av = 10.0, bv = 3.0;
	TensorHandle a = tensor_create_scalar_streamed(av, 0, 0, 14);
	TensorHandle b = tensor_create_scalar_streamed(bv, 0, 0, 14);
	TensorHandle c = tensor_sub(a, b);
	cr_assert_str_eq(tensor_dtype_name(c), "F32", "F32 sub output keeps F32 tag");
	cr_assert_float_eq(tensor_item(c), 7.0, 1e-5);
}

Test(core_elementwise_sub, f32_forward_vector_same_shape) {
	/* Same-shape F32 vDSP path (fn_sub_f32 stamping). */
	double ad[] = {10.0, 20.0, 30.0};
	double bd[] = {1.0, 2.0, 3.0};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(ad, 3), 0, 0, 14);
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(bd, 3), 0, 0, 14);
	TensorHandle c = tensor_sub(a, b);
	cr_assert_str_eq(tensor_dtype_name(c), "F32");
	double out[3];
	tensor_to_doubles(c, out);
	cr_assert_float_eq(out[0], 9.0, 1e-5);
	cr_assert_float_eq(out[1], 18.0, 1e-5);
	cr_assert_float_eq(out[2], 27.0, 1e-5);
}

Test(core_elementwise_sub, f32_backward_vector_same_shape) {
	/* F32 same-shape backward: d(sum(a-b))/da = +1, /db = -1 (sign flip). */
	param_clear();
	double ad[] = {5.0, 6.0, 7.0};
	double bd[] = {1.0, 2.0, 3.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 14);
	TensorHandle b = tensor_create_param_1d_streamed(3, hcopy(bd, 3), 0, 14);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5, "d(sum(a-b))/da[%d] should be +1.0",
		                   i);
		cr_assert_float_eq(param_grad_item_at(1, i), -1.0, 1e-5,
		                   "d(sum(a-b))/db[%d] should be -1.0", i);
	}
	param_clear();
}

#ifdef BACKEND_TAPE
/* Death test for sub.c line 25's mixed-dtype guard call. */
Test(core_elementwise_sub, mixed_dtype_aborts, .signal = SIGABRT) {
	double av = 1.0, bv = 2.0;
	TensorHandle a = tensor_create_scalar_streamed(av, 0, 0, 14); /* F32 */
	TensorHandle b = tensor_create_scalar(bv, 0);                 /* F64 */
	(void)tensor_sub(a, b);                                       /* must abort */
}
#endif /* BACKEND_TAPE */

/* DISABLED: tape general-broadcast elementwise crash — see TODO.md. */
Test(core_elementwise_sub, forward_general_broadcast_row, .disabled = true) {
	/* [2,3] - [3] broadcasts across rows; exercises compute_bcast_shape. */
	double ad[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
	TensorHandle a = tensor_create_2d(2, 3, ad, 0);
	double bd[] = {1.0, 2.0, 3.0};
	int bs[] = {3};
	TensorHandle b = tensor_create(bd, bs, 1, 0);
	TensorHandle c = tensor_sub(a, b);
	cr_assert_eq(tensor_dim(c), 2);
	cr_assert_eq(tensor_size(c, 0), 2);
	cr_assert_eq(tensor_size(c, 1), 3);
	double out[6];
	tensor_to_doubles(c, out);
	double expected[] = {9.0, 18.0, 27.0, 39.0, 48.0, 57.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expected[i], 1e-12, "broadcast sub out[%d]", i);
}

/* DISABLED: tape general-broadcast elementwise crash — see TODO.md. */
Test(core_elementwise_sub, backward_general_broadcast_row, .disabled = true) {
	/* c = a[2,3] - b[3]; loss = sum(c). Exercises the general numpy-broadcast
	   backward path (sub.c lines 58-85) + compute_bcast_strides.
	   d_a[i] = +1; b broadcasts across 2 rows with sign flip => d_b[j] = -2. */
	param_clear();
	double ad[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
	TensorHandle a = tensor_create_2d(2, 3, ad, 1);
	double bd[] = {1.0, 2.0, 3.0};
	int bs[] = {3};
	TensorHandle b = tensor_create(bd, bs, 1, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "d(sum(a-b))/da[%d] should be +1.0", i);
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), -2.0, 1e-12,
		                   "d(sum(a-b))/db[%d] should be -2.0 (summed over 2 rows, sign-flipped)",
		                   j);
}

/* DISABLED: tape general-broadcast elementwise crash — see TODO.md. */
Test(core_elementwise_sub, backward_general_broadcast_col, .disabled = true) {
	/* c = a[2,3] - b[2,1]; b broadcasts across 3 columns, sign-flipped =>
	   d_b[i] = -3.0. Different stride pattern (trailing size-1 axis). */
	param_clear();
	double ad[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
	TensorHandle a = tensor_create_2d(2, 3, ad, 1);
	double bd[] = {1.0, 2.0};
	TensorHandle b = tensor_create_2d(2, 1, bd, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_sub(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "d(sum(a-b))/da[%d] should be +1.0", i);
	for (int i = 0; i < 2; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), -3.0, 1e-12,
		                   "d(sum(a-b))/db[%d] should be -3.0 (summed over 3 cols, sign-flipped)",
		                   i);
}

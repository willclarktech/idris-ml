/* Criterion suite for tape `tensor_gather`. */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

/* Streamed creators FREE their data argument (callee-owns) — hand each one a
   fresh heap copy so the caller's stack buffer is never freed or aliased. */
static double* hcopy(const double* s, int n) {
	double* b = malloc((size_t)n * sizeof(double));
	memcpy(b, s, (size_t)n * sizeof(double));
	return b;
}

Test(linear_index_gather, forward_with_index) {
	/* input = [10, 20, 30, 40], index = [3, 1, 0] -> [40, 20, 10] */
	double id[] = {10.0, 20.0, 30.0, 40.0};
	double ixd[] = {3.0, 1.0, 0.0};
	int s_in[] = {4};
	int s_ix[] = {3};
	TensorHandle input = tensor_create(id, s_in, 1, 0);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	TensorHandle r = tensor_gather(input, index, 3);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 40.0, 1e-12);
	cr_assert_float_eq(out[1], 20.0, 1e-12);
	cr_assert_float_eq(out[2], 10.0, 1e-12);
}

Test(linear_index_gather, backward_accumulates_duplicate_index) {
	/* index = [1, 1, 0] picks position 1 twice; backward scatter-adds, so
	   d_input[1] accumulates 2 and d_input[0] = 1, d_input[2,3] = 0. */
	param_clear();
	double id[] = {10.0, 20.0, 30.0, 40.0};
	double ixd[] = {1.0, 1.0, 0.0};
	int s_in[] = {4};
	int s_ix[] = {3};
	TensorHandle input = tensor_create(id, s_in, 1, 1);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	param_register("input", input);
	TensorHandle r = tensor_gather(input, index, 3);
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 20.0, 1e-12);
	cr_assert_float_eq(out[1], 20.0, 1e-12);
	cr_assert_float_eq(out[2], 10.0, 1e-12);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 1), 2.0, 1e-12, "input grad[1] should be 2 (got %.6f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 3), 0.0, 1e-12);
}

Test(linear_index_gather, backward_scatters_grad) {
	/* gather with index [3, 1, 0]; sum -> d_input scattered to [3,1,0]. */
	param_clear();
	double id[] = {10.0, 20.0, 30.0, 40.0};
	double ixd[] = {3.0, 1.0, 0.0};
	int s_in[] = {4};
	int s_ix[] = {3};
	TensorHandle input = tensor_create(id, s_in, 1, 1);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	param_register("input", input);
	TensorHandle r = tensor_gather(input, index, 3);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	/* d_input should be [1, 1, 0, 1] — positions 0, 1, 3 picked, 2 unpicked */
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12, "input grad[1] should be 1 (got %.6f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12);
}

/* ---- F32 coverage (gather.c lines 21-24, 26-27: F32 arena output) ---- */

Test(linear_index_gather, f32_forward_with_index) {
	/* F32 input = [10, 20, 30, 40], index = [3, 1, 0] -> [40, 20, 10].
	   Index tensor stays F64 (tape_load_d is dtype-agnostic); the input
	   being F32 selects the F32 arena path. F32 readback ~1e-6 error. */
	double id[] = {10.0, 20.0, 30.0, 40.0};
	double ixd[] = {3.0, 1.0, 0.0};
	int s_ix[] = {3};
	TensorHandle input = tensor_create_1d_streamed(4, hcopy(id, 4), 0, 0, 14);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	cr_assert_str_eq(tensor_dtype_name(input), "F32");
	TensorHandle r = tensor_gather(input, index, 3);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 gather output keeps F32 tag");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 40.0, 1e-5);
	cr_assert_float_eq(out[1], 20.0, 1e-5);
	cr_assert_float_eq(out[2], 10.0, 1e-5);
}

Test(linear_index_gather, f32_backward_scatters_grad) {
	/* F32 input requires_grad; gather index [3,1,0], sum -> d_input = [1,1,0,1].
	   Exercises the F32 forward arena path + tape append; the backward
	   (tape_backward_gather) is dtype-agnostic. */
	param_clear();
	double id[] = {10.0, 20.0, 30.0, 40.0};
	double ixd[] = {3.0, 1.0, 0.0};
	int s_ix[] = {3};
	TensorHandle input = tensor_create_param_1d_streamed(4, hcopy(id, 4), 0, 14);
	TensorHandle index = tensor_create(ixd, s_ix, 1, 0);
	param_register("input", input);
	TensorHandle r = tensor_gather(input, index, 3);
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-5);
	cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-5, "input grad[1] should be 1 (got %.6f)",
	                   param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(0, 2), 0.0, 1e-5);
	cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-5);
	param_clear();
}

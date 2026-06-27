/* Criterion suites for tape `tensor_max_pool1d`.
 *
 * conv_max_pool1d: F64 forward/backward (winner subgradient routing).
 * max_pool1d_f32_cov: F32 storage/kernel arms via the streamed dtag-14 creator.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"
#include "port_assert.h"

Test(conv_max_pool1d, forward_and_backward) {
	param_clear();
	double in_data[4] = {3.0, 1.0, 4.0, 2.0};
	int sh[2] = {1, 4};
	TensorHandle in = tensor_create(in_data, sh, 2, 1);
	param_register("in", in);

	TensorHandle out = tensor_max_pool1d(in, /*kL=*/2, /*stride=*/2);
	cr_assert_float_eq(tensor_item_1d(out, 0), 3.0, 1e-12);
	cr_assert_float_eq(tensor_item_1d(out, 1), 4.0, 1e-12);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	/* Winners: idx 0 and idx 2 each get 1.0; idx 1 and 3 get 0. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12, "d_in[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, 1e-12, "d_in[1]");
	cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "d_in[2]");
	cr_assert_float_eq(param_grad_item_at(0, 3), 0.0, 1e-12, "d_in[3]");
}

#ifdef BACKEND_TAPE

/* C=2, L=4, kL=2, stride=2 -> oL = (4-2)/2+1 = 2. Non-overlapping windows.
   row0 = [1,5,3,2]: ol0 win[1,5]->5 (flat 1); ol1 win[3,2]->3 (flat 2).
   row1 = [9,4,6,8]: ol0 win[9,4]->9 (flat 4); ol1 win[6,8]->8 (flat 7).
   out = [5,3,9,8]. Drives the F32 forward write (line 41) +
   make_tensor_arena_f32 result arm (line 48). */
Test(max_pool1d_f32_cov, f32_forward_backward) {
	param_clear();
	double in_data[8] = {1.0, 5.0, 3.0, 2.0, 9.0, 4.0, 6.0, 8.0};
	TensorHandle in = tensor_create_2d_streamed(2, 4, hcopy(in_data, 8), 1, 0, 14);
	param_register("in", in);

	TensorHandle out = tensor_max_pool1d(in, 2, 2);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 4);
	double od[4];
	tensor_to_doubles(out, od);
	double exp_out[4] = {5.0, 3.0, 9.0, 8.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(od[i], exp_out[i], TEST_TOL_RELAXED, "out[%d] should be %.1f (got %.6f)",
		                   i, exp_out[i], od[i]);

	/* sum-loss: subgradient routes 1.0 to each winning input slot. */
	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	double exp_din[8] = {0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), exp_din[i], TEST_TOL_RELAXED,
		                   "d_in[%d] should be %.1f (got %.6f)", i, exp_din[i],
		                   param_grad_item_at(0, i));

	param_clear();
}

/* Overlapping windows (stride < kL): C=1, L=4, kL=3, stride=1 ->
   oL = (4-3)/1+1 = 2. in = [2,7,1,5].
   ol0 win[2,7,1]->7 (flat 1); ol1 win[7,1,5]->7 (flat 1 again — same winner).
   out = [7,7]. Both outputs route their d_out to flat 1, so d_in[1]=2.
   Exercises the F32 forward write + a winner shared across outputs. */
Test(max_pool1d_f32_cov, f32_overlap_shared_winner) {
	param_clear();
	double in_data[4] = {2.0, 7.0, 1.0, 5.0};
	TensorHandle in = tensor_create_2d_streamed(1, 4, hcopy(in_data, 4), 1, 0, 14);
	param_register("in", in);

	TensorHandle out = tensor_max_pool1d(in, 3, 1);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 2);
	double od[2];
	tensor_to_doubles(out, od);
	cr_assert_float_eq(od[0], 7.0, TEST_TOL_RELAXED, "out[0] should be 7 (got %.6f)", od[0]);
	cr_assert_float_eq(od[1], 7.0, TEST_TOL_RELAXED, "out[1] should be 7 (got %.6f)", od[1]);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	double exp_din[4] = {0.0, 2.0, 0.0, 0.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), exp_din[i], TEST_TOL_RELAXED,
		                   "d_in[%d] should be %.1f (got %.6f)", i, exp_din[i],
		                   param_grad_item_at(0, i));

	param_clear();
}

#endif /* BACKEND_TAPE */

Test(conv_max_pool1d, max_pool1d_forward) {
	double inp_data[] = {1, 3, 2, 4, 5, 1};
	int inp_shape[] = {1, 6};
	TensorHandle inp = tensor_create(inp_data, inp_shape, 2, 0);

	TensorHandle out = tensor_max_pool1d(inp, 2, 2);
	ASSERT_TRUE("pool1d size1", tensor_size(out, 1) == 3);
	double result[3];
	tensor_to_doubles(out, result);
	ASSERT_NEAR("pool1d[0]", result[0], 3.0, 1e-10);
	ASSERT_NEAR("pool1d[1]", result[1], 4.0, 1e-10);
	ASSERT_NEAR("pool1d[2]", result[2], 5.0, 1e-10);
}

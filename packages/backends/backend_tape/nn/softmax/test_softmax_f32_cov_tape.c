/* Criterion suite `softmax_f32_cov` — coverage top-up for tape softmax.c.
 *
 * The pre-existing test_softmax_cov_tape.c exercises the F64 paths. This
 * file closes the F32 scalar (rank==0) store arm of tensor_softmax_f32
 * (softmax.c lines 36-37: make_scalar_f32 in the rank==0 branch), reached
 * only with an F32-tagged scalar input routed through tensor_softmax.
 *
 * Oracle: softmax over a single element is exp(x-x)/sum = 1/1 = 1.0,
 * independent of the input value. Because the output is the constant 1,
 * the input gradient is exactly 0 (d r/d x = sm*(1-sm) = 1*0 = 0).
 *
 * F32-tagged tensors store as float, so reads use TEST_TOL_RELAXED; the
 * chosen value (5) is integer-exact in single precision.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* F32 scalar forward: drives the rank==0 make_scalar_f32 store arm. */
Test(softmax_f32_cov, scalar_f32_forward) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(5.0, 0, 0, 14);
	TensorHandle r = tensor_softmax(x, 0);
	cr_assert_eq(tensor_numel(r), 1);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 scalar -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_float_eq(tensor_item_1d(r, 0), 1.0, TEST_TOL_RELAXED,
	                   "softmax of a scalar should be 1.0 (got %.6f)", tensor_item_1d(r, 0));
	param_clear();
}

/* F32 scalar forward + backward: same store arm with requires_grad set, so
   the tape-append branch fires too. The constant-1 output makes the input
   gradient exactly 0. */
Test(softmax_f32_cov, scalar_f32_backward) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(5.0, 1, 0, 14);
	param_register("x", x);
	TensorHandle r = tensor_softmax(x, 0);
	cr_assert_eq(tensor_numel(r), 1);
	cr_assert_float_eq(tensor_item_1d(r, 0), 1.0, TEST_TOL_RELAXED,
	                   "softmax of a scalar should be 1.0 (got %.6f)", tensor_item_1d(r, 0));
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TEST_TOL_RELAXED,
	                   "d softmax(scalar) / d x should be 0 (got %.6f)", param_grad_item_at(0, 0));
	param_clear();
}

#endif /* BACKEND_TAPE */

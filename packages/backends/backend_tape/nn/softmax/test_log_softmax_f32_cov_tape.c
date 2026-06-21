/* Criterion suite `log_softmax_f32_cov` — coverage top-up for the tape
 * log_softmax source (nn/softmax/log_softmax.c).
 *
 * The pre-existing log_softmax tests exercise the F64 and F32 rank>=1
 * (vector) paths. This file closes the remaining uncovered rank==0
 * (scalar) arms of tensor_log_softmax:
 *
 *   - lines 29-30: the F32 scalar branch (make_scalar_f32), reached only
 *                  when a DT_F32-tagged tensor has rank 0.
 *   - line 57    : the F64 scalar branch (make_scalar), reached only when
 *                  an F64 tensor has rank 0.
 *
 * Oracle: log_softmax over a single element is identically 0. The stable
 * max-subtract makes the exponent 0, sum = exp(0) = 1, log_sum = log(1) +
 * max = max, so r = x - log_sum = x - max = 0 regardless of the input
 * value. Backward: with loss = sum(r), grad(r[0]) = 1, and
 * d_x[0] = grad[0] - exp(r[0]) * sum_grad = 1 - exp(0)*1 = 0.
 *
 * F32 storage reads use TEST_TOL_RELAXED; the oracle (0) is exact in both
 * precisions.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* F32 scalar: drives tensor_log_softmax_f32's rank==0 arm (make_scalar_f32,
   lines 29-30). The streamed dtag-14 creator owns+frees its (scalar) value. */
Test(log_softmax_f32_cov, scalar_f32) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(3.5, 1, 0, 14);
	param_register("x", x);
	TensorHandle r = tensor_log_softmax(x, 0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 scalar -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 1);
	cr_assert_float_eq(tensor_item(r), 0.0, TEST_TOL_RELAXED,
	                   "log_softmax of a scalar is 0 (got %.6f)", tensor_item(r));
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TEST_TOL_RELAXED,
	                   "d_x[0] should be 0 (got %.6f)", param_grad_item_at(0, 0));
	param_clear();
}

/* F64 scalar: drives tensor_log_softmax's rank==0 arm (make_scalar, line 57). */
Test(log_softmax_f32_cov, scalar_f64) {
	param_clear();
	TensorHandle x = tensor_create_scalar(-2.0, 1);
	param_register("x", x);
	TensorHandle r = tensor_log_softmax(x, 0);
	cr_assert_str_eq(tensor_dtype_name(r), "F64", "F64 scalar -> F64 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 1);
	cr_assert_float_eq(tensor_item(r), 0.0, TEST_TOL_TIGHT,
	                   "log_softmax of a scalar is 0 (got %.9f)", tensor_item(r));
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TEST_TOL_TIGHT,
	                   "d_x[0] should be 0 (got %.9f)", param_grad_item_at(0, 0));
	param_clear();
}

#endif /* BACKEND_TAPE */

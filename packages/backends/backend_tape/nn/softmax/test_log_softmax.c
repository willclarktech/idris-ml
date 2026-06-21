/* Criterion coverage for the tape log_softmax source (nn/softmax/log_softmax.c).
 *
 * Covers the rank==0 (scalar) arms of tensor_log_softmax for both the F32
 * (make_scalar_f32) and F64 (make_scalar) paths. Oracle: log_softmax over a
 * single element is identically 0 (stable max-subtract -> exp(0)=1,
 * log_sum=max, r=x-max=0); backward d_x[0] = 1 - exp(0)*1 = 0.
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

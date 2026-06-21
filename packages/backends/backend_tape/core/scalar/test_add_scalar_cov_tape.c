/* add_scalar F32 arm coverage (tape). tensor_add_scalar's DT_F32 branch
 * (tensor_add_scalar_f32) is reachable only with an F32 input; built via the
 * streamed dtag-14 path (tape's bare F32 creators abort). */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

Test(add_scalar_cov, f32_forward_backward) {
	param_clear();
	double d[] = {1.0, 2.0, 3.0};
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(d, 3), /*rg=*/1, /*stream_tag=*/0, 14);
	param_register("x", x);
	TensorHandle y = tensor_add_scalar(x, 5.0); /* F32 rank>=1 loop */
	double out[3];
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 6.0, TEST_TOL_RELAXED, "x+5 [0] (got %.6f)", out[0]);
	cr_assert_float_eq(out[2], 8.0, TEST_TOL_RELAXED, "x+5 [2] (got %.6f)", out[2]);
	TensorHandle loss = tensor_sum(y);
	tensor_backward(loss); /* d(x+s)/dx = 1 */
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_RELAXED,
		                   "d(x+5)/dx[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	param_clear();
}

/* numel==1 F32 scalar: hits the make_scalar_f32 fast-path arm (a->numel == 1). */
Test(add_scalar_cov, f32_scalar_fastpath) {
	param_clear();
	TensorHandle x = tensor_create_scalar_streamed(3.0, /*rg=*/1, /*stream_tag=*/0, 14);
	param_register("x", x);
	TensorHandle y = tensor_add_scalar(x, 5.0);
	cr_assert_float_eq(tensor_item(y), 8.0, TEST_TOL_RELAXED, "3+5 (got %.6f)", tensor_item(y));
	tensor_backward(y);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_RELAXED, "d(x+5)/dx = 1 (got %.6f)",
	                   param_grad_item_at(0, 0));
	param_clear();
}

#endif /* BACKEND_TAPE */

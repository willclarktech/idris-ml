/* mul_scalar F32 arm coverage (tape). tensor_mul_scalar's DT_F32 branch
 * (tensor_mul_scalar_f32) is reachable only with an F32 input; tape's bare
 * F32 creators abort, so build via the streamed dtag-14 path. */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

Test(mul_scalar_cov, f32_forward_backward) {
	param_clear();
	double d[] = {1.0, 2.0, 3.0};
	TensorHandle x = tensor_create_1d_streamed(3, hcopy(d, 3), /*rg=*/1, /*stream_tag=*/0, 14);
	param_register("x", x);
	TensorHandle y = tensor_mul_scalar(x, 2.0); /* F32 rank>=1 loop */
	double out[3];
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 2.0, TEST_TOL_RELAXED, "x*2 [0] (got %.6f)", out[0]);
	cr_assert_float_eq(out[2], 6.0, TEST_TOL_RELAXED, "x*2 [2] (got %.6f)", out[2]);
	TensorHandle loss = tensor_sum(y);
	tensor_backward(loss); /* d(x*s)/dx = s = 2 */
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 2.0, TEST_TOL_RELAXED,
		                   "d(x*2)/dx[%d] should be 2 (got %.6f)", i, param_grad_item_at(0, i));
	param_clear();
}

#endif /* BACKEND_TAPE */

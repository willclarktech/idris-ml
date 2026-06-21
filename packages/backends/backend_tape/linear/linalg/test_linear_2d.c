/* Criterion coverage for tensor_linear_2d — the F32 zero-dim
 * bias-broadcast arm (linear_2d.c lines 41-44).
 *
 * Tape-only: the streamed dtag-14 creation path + the float-output oracle
 * are tape-specific, so the whole file is wrapped in #ifdef BACKEND_TAPE
 * (it sits in a tape dir but is compiled into every backend's test binary).
 */

#ifdef BACKEND_TAPE

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

Test(linear_2d_f32_extra_cov, zero_dim_i0_bias_f32) {
	/* W=[2,0], X=[2,0], bias=[10,20] — all F32 (dtag 14). With ii==0 the
	   matmul drops out and the bias broadcasts across the batch into an
	   F32 output: Y[b,o] = bias[o], so Y = [[10,20],[10,20]] flattened to
	   {10,20,10,20}. Covers the DT_F32 ii==0 bias-broadcast arm
	   (linear_2d.c lines 41-44). requires_grad=0 throughout (mirrors the
	   F64 zero-dim test — the guard never reaches the tape-append path). */
	double wd[] = {0};
	double xd[] = {0};
	double bd[] = {10.0, 20.0};
	/* W=[2,0] and X=[2,0] have numel 0; the streamed creators own+free the
	   (zero-length) heap copy. bias=[2] carries the broadcast values. */
	TensorHandle W = tensor_create_2d_streamed(2, 0, hcopy(wd, 0), 0, 0, 14);
	TensorHandle X = tensor_create_2d_streamed(2, 0, hcopy(xd, 0), 0, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(2, hcopy(bd, 2), 0, 0, 14);
	TensorHandle r = tensor_linear_2d(W, X, bias);
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected_Y[] = {10, 20, 10, 20};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected_Y[i], TEST_TOL_RELAXED,
		                   "Y_f32[%d] should be broadcast bias %.1f (got %.6f)", i, expected_Y[i],
		                   out[i]);
}

#endif /* BACKEND_TAPE */

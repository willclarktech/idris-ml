/* Criterion suite `linear_f32_extra_cov` — tape-only top-up for linear.c.
 *
 * Closes the F32 zero-dim (n==0) bias-only branch of tensor_linear_f32
 * (linear.c lines 35-36): when the inner dim collapses, the matmul drops
 * out and the output is just the (float) bias copied through. The existing
 * `linalg_cov` suite covers only the F64 equivalent (linear.c 93-95).
 *
 * NOT covered here (deliberately): linear.c line 84,
 * `tape_abort_mixed_dtype("tensor_linear")`, is a noreturn abort guard —
 * reaching it requires a death test, which is out of scope for this file.
 *
 * F32-tagged tensors store as float, so readbacks use TEST_TOL_RELAXED.
 * All values are exact integers representable in single precision.
 *
 * Tape-only: F32 here is the streamed dtag-14 creation path, and the
 * zero-tensor + float-bias-copy oracle is tape-specific. The file lives in
 * a tape dir but is compiled into every backend's test binary, so the whole
 * body is wrapped in #ifdef BACKEND_TAPE.
 */

#ifdef BACKEND_TAPE

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

Test(linear_f32_extra_cov, linear_f32_zero_dim_n0_bias) {
	/* W=[2,0] (F32), x=[0] (F32), bias=[10,20] (F32): the n==0 guard inside
	   tensor_linear_f32 returns a zero F32 tensor of shape [m]=[2] and copies
	   the float bias into it. Output = bias = [10, 20]. Covers lines 35-36. */
	param_clear();
	double wd[] = {0};
	double xd[] = {0};
	double bd[] = {10.0, 20.0};
	/* Streamed dtag-14 creators OWN+free their buffer; pass hcopy(...).
	   rg=0: no tape append happens in the zero-dim guard, so no backward. */
	TensorHandle W = tensor_create_2d_streamed(2, 0, hcopy(wd, 0), 0, 0, 14);
	TensorHandle x = tensor_create_1d_streamed(0, hcopy(xd, 0), 0, 0, 14);
	TensorHandle bias = tensor_create_1d_streamed(2, hcopy(bd, 2), 0, 0, 14);
	TensorHandle r = tensor_linear(W, x, bias);
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 10.0, TEST_TOL_RELAXED, "y_f32[0] should be bias 10 (got %.6f)",
	                   out[0]);
	cr_assert_float_eq(out[1], 20.0, TEST_TOL_RELAXED, "y_f32[1] should be bias 20 (got %.6f)",
	                   out[1]);
	param_clear();
}

#endif /* BACKEND_TAPE */

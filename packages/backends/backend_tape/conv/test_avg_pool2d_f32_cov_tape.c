/* Criterion suite `avg_pool2d_f32_cov` — coverage top-up for the tape
 * conv/avg_pool2d.c F32 arms.
 *
 * The base coverage only exercises the F64 storage/kernel path of
 * tensor_avg_pool2d. This file closes the two uncovered F32 arms in the
 * forward op (both gated by `is_f32 == (dtype_tag == DT_F32)`):
 *
 *   - line 35: the F32 store of the windowed average into the float
 *     output buffer (`((float*)out)[...] = (float)v`);
 *   - line 41: the F32 result construction via make_tensor_arena_f32.
 *
 * Both arms are reached purely by calling tensor_avg_pool2d on an F32
 * (streamed dtag-14) rank-3 input; the backward op itself is dtype-
 * agnostic (tape_grad_*_d) so a sum-loss backward additionally checks the
 * gradient flows correctly through the F32 forward.
 *
 * Oracles are computed by hand. All chosen averages (2.5, 6.5, 1.0) and
 * the per-element gradient (scale = 1/(kH*kW) = 0.25) are exactly
 * representable in single precision, so TEST_TOL_RELAXED is generous.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE
/* Dtag value mirroring DType.Core ("13/14/15=F16/F32/F64"). */
#define DTAG_F32 14

/* Input [C=1, H=2, W=2] = {1,2,3,4}, kH=kW=2, stride=2 -> oH=oW=1.
   Single 2x2 window: avg = (1+2+3+4)/4 = 2.5.
   Hits the F32 store (line 35) + make_tensor_arena_f32 (line 41). */
Test(avg_pool2d_f32_cov, forward_f32_single_window) {
	param_clear();
	double in_src[4] = {1.0, 2.0, 3.0, 4.0};
	int sh_in[3] = {1, 2, 2};
	TensorHandle in = tensor_create_streamed(hcopy(in_src, 4), sh_in, 3, 0, 0, DTAG_F32);

	TensorHandle out = tensor_avg_pool2d(in, 2, 2, 2, 2);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 1);
	double o[1];
	tensor_to_doubles(out, o);
	cr_assert_float_eq(o[0], 2.5, TEST_TOL_RELAXED, "avg should be 2.5 (got %.6f)", o[0]);
	param_clear();
}

/* Input [C=2, H=2, W=2], kH=kW=2, stride=2 -> oH=oW=1, numel=2.
   ch0 = {1,2,3,4} -> avg 2.5; ch1 = {5,6,7,8} -> avg 6.5.
   Exercises the F32 forward store across the C loop (line 35) + the F32
   result construction (line 41), then a sum-loss backward: each input
   element contributes to exactly one window, so d_in[i] = scale = 0.25. */
Test(avg_pool2d_f32_cov, forward_backward_f32_multichannel) {
	param_clear();
	double in_src[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	int sh_in[3] = {2, 2, 2};
	TensorHandle in = tensor_create_streamed(hcopy(in_src, 8), sh_in, 3, 1, 0, DTAG_F32);
	param_register("in", in);

	TensorHandle out = tensor_avg_pool2d(in, 2, 2, 2, 2);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 2);
	double o[2];
	tensor_to_doubles(out, o);
	cr_assert_float_eq(o[0], 2.5, TEST_TOL_RELAXED, "ch0 avg should be 2.5 (got %.6f)", o[0]);
	cr_assert_float_eq(o[1], 6.5, TEST_TOL_RELAXED, "ch1 avg should be 6.5 (got %.6f)", o[1]);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	/* scale = 1/(2*2) = 0.25; non-overlapping windows -> one window each. */
	for (int i = 0; i < 8; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.25, TEST_TOL_RELAXED,
		                   "d_in[%d] should be 0.25", i);
	param_clear();
}

/* Input [C=1, H=4, W=4] all ones, kH=kW=2, stride=2 -> oH=oW=2, numel=4.
   Every 2x2 window averages 1.0; non-overlapping so d_in[i] = 0.25.
   Exercises the oH/oW output loops (line 35) with a multi-position output. */
Test(avg_pool2d_f32_cov, forward_backward_f32_multiwindow) {
	param_clear();
	double in_src[16];
	for (int i = 0; i < 16; i++)
		in_src[i] = 1.0;
	int sh_in[3] = {1, 4, 4};
	TensorHandle in = tensor_create_streamed(hcopy(in_src, 16), sh_in, 3, 1, 0, DTAG_F32);
	param_register("in", in);

	TensorHandle out = tensor_avg_pool2d(in, 2, 2, 2, 2);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(out));
	cr_assert_eq(tensor_numel(out), 4);
	double o[4];
	tensor_to_doubles(out, o);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(o[i], 1.0, TEST_TOL_RELAXED, "out[%d] avg should be 1.0 (got %.6f)", i,
		                   o[i]);

	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	for (int i = 0; i < 16; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.25, TEST_TOL_RELAXED,
		                   "d_in[%d] should be 0.25", i);
	param_clear();
}
#endif /* BACKEND_TAPE */

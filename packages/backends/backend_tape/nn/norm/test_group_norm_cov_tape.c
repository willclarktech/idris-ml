/* Criterion suite `group_norm_cov` — coverage top-up for tape group_norm.c.
 *
 * The base coverage only exercises the F64 forward path. This file closes
 * the remaining uncovered F32 arms of `tensor_group_norm` (streamed
 * dtag-14 inputs):
 *
 *   - line 56: the `((float*)out)[idx] = (float)v;` store branch (is_f32);
 *   - line 64: the `make_tensor_arena_f32(...)` F32 return branch.
 *
 * group_norm is forward-only (no backward tape entry — "torch/MLX handle
 * group norm natively"), so these are forward-only tests. Line 30 (the
 * tape_abort_mixed_dtype guard) is an abort path and is intentionally NOT
 * covered here (death tests for abort guards are excluded).
 *
 * Oracles are hand-computed. All cases use eps=0 with a variance that is a
 * perfect square (so rstd is exact in F32) and integer/half affine params,
 * making every normalized output exactly representable in single precision.
 *
 * Tape-only: F32 on tape requires the streamed dtag-14 creators (bare
 * tensor_create_*_f32 aborts), and the oracle here assumes tape's exact
 * F64-internal accumulation, so the whole file is BACKEND_TAPE-gated.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE
/* Dtag value mirroring DType.Core ("14 = F32"). */
#define DTAG_F32 14

/* channels=2, spatial=1, numGroups=1: one group over both channels.
   input=[1,3] -> mean=2, var=((1-2)^2+(3-2)^2)/2=1, rstd=1/sqrt(1)=1.
   x_hat=[-1,1]. affine gamma=[2,4], beta=[10,20]:
     out[0]=2*(-1)+10=8 ; out[1]=4*1+20=24.
   Hits the F32 store (line 56) + F32 return (line 64). */
Test(group_norm_cov, f32_single_group_affine) {
	double in_d[] = {1.0, 3.0};
	double g_d[] = {2.0, 4.0};
	double b_d[] = {10.0, 20.0};
	TensorHandle in = tensor_create_1d_streamed(2, hcopy(in_d, 2), 0, 0, DTAG_F32);
	TensorHandle gamma = tensor_create_1d_streamed(2, hcopy(g_d, 2), 0, 0, DTAG_F32);
	TensorHandle beta = tensor_create_1d_streamed(2, hcopy(b_d, 2), 0, 0, DTAG_F32);
	TensorHandle r = tensor_group_norm(in, gamma, beta, 1, 2, 1, 0.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 2);
	double out[2];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 8.0, TEST_TOL_RELAXED, "gn_f32[0] should be 8 (got %.6f)", out[0]);
	cr_assert_float_eq(out[1], 24.0, TEST_TOL_RELAXED, "gn_f32[1] should be 24 (got %.6f)", out[1]);
}

/* channels=4, spatial=1, numGroups=2: chPerGroup=2, groupSize=2.
   group0=[1,3]: mean=2, var=1, rstd=1 -> x_hat=[-1,1].
   group1=[5,9]: mean=7, var=((5-7)^2+(9-7)^2)/2=4, rstd=1/sqrt(4)=0.5
                 -> x_hat=[-1,1].
   gamma=all 1, beta=all 0 -> out=[-1,1,-1,1]. Exercises numGroups>1 and
   a non-unit rstd (0.5 is exact in F32). */
Test(group_norm_cov, f32_two_groups) {
	double in_d[] = {1.0, 3.0, 5.0, 9.0};
	double g_d[] = {1.0, 1.0, 1.0, 1.0};
	double b_d[] = {0.0, 0.0, 0.0, 0.0};
	TensorHandle in = tensor_create_1d_streamed(4, hcopy(in_d, 4), 0, 0, DTAG_F32);
	TensorHandle gamma = tensor_create_1d_streamed(4, hcopy(g_d, 4), 0, 0, DTAG_F32);
	TensorHandle beta = tensor_create_1d_streamed(4, hcopy(b_d, 4), 0, 0, DTAG_F32);
	TensorHandle r = tensor_group_norm(in, gamma, beta, 2, 4, 1, 0.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double exp[] = {-1.0, 1.0, -1.0, 1.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], exp[i], TEST_TOL_RELAXED,
		                   "gn2_f32[%d] should be %.1f (got %.6f)", i, exp[i], out[i]);
}

/* channels=2, spatial=2, numGroups=1: n=4, groupSize=4. Exercises the inner
   spatial loop with spatial>1. Flat layout [c0s0,c0s1,c1s0,c1s1].
   input=[1,-1,1,-1] -> mean=0, var=(1+1+1+1)/4=1, rstd=1 -> x_hat=input.
   gamma=all 1, beta=all 0 -> out=input=[1,-1,1,-1]. */
Test(group_norm_cov, f32_spatial2) {
	double in_d[] = {1.0, -1.0, 1.0, -1.0};
	double g_d[] = {1.0, 1.0};
	double b_d[] = {0.0, 0.0};
	TensorHandle in = tensor_create_1d_streamed(4, hcopy(in_d, 4), 0, 0, DTAG_F32);
	TensorHandle gamma = tensor_create_1d_streamed(2, hcopy(g_d, 2), 0, 0, DTAG_F32);
	TensorHandle beta = tensor_create_1d_streamed(2, hcopy(b_d, 2), 0, 0, DTAG_F32);
	TensorHandle r = tensor_group_norm(in, gamma, beta, 1, 2, 2, 0.0);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 inputs -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double exp[] = {1.0, -1.0, 1.0, -1.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], exp[i], TEST_TOL_RELAXED,
		                   "gnsp_f32[%d] should be %.1f (got %.6f)", i, exp[i], out[i]);
}
#endif /* BACKEND_TAPE */

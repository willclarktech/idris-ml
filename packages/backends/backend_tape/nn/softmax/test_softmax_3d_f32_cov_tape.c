/* Criterion suite `softmax_3d_f32_cov` — coverage top-up for the tape
 * nn/softmax/softmax_3d.c F32 forward arm.
 *
 * The base coverage exercises the F64 path. This file closes the three
 * uncovered F32 lines of tensor_softmax_3d, all inside the single
 * `is_f32` branch:
 *   - line 29: ((float*)data)[..] = (float)e;   (F32 store in the exp loop)
 *   - line 36: ((float*)data)[..] /= (float)sum; (F32 normalization)
 *   - line 44: make_tensor_arena_f32(...)        (F32 result construction)
 *
 * F32 tensors on tape MUST be built via the streamed dtag-14 creators
 * (bare *_f32 creators abort); tensor_create_streamed OWNS+frees its
 * buffer, so it gets hcopy(...). Oracles are hand-computed and chosen so
 * the softmax outputs are exact (or near-exact) in single precision:
 * equal-valued rows give 1/n, and the [0, ln3] row gives [1/4, 3/4].
 */

#include <criterion/criterion.h>
#include <math.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE
/* Dtag mirroring DType.Core ("13/14/15=F16/F32/F64"). */
#define DTAG_F32 14

/* [B,m,n] = [2,1,2], every element equal (1.0). Each length-2 row softmaxes
   to [0.5, 0.5] — exact in F32. Drives the F32 store/normalize/create arm.
   Also runs the backward: for a uniform softmax under a sum-loss, dot==1 per
   row so every input grad is exactly 0 (line 49 tape_append + backward). */
Test(softmax_3d_f32_cov, f32_forward_backward_uniform) {
	param_clear();
	double in[4] = {1.0, 1.0, 1.0, 1.0};
	int sh[3] = {2, 1, 2};
	TensorHandle x = tensor_create_streamed(hcopy(in, 4), sh, 3, 1, 0, DTAG_F32);
	param_register("x", x);

	TensorHandle r = tensor_softmax_3d(x);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], 0.5, TEST_TOL_RELAXED, "softmax[%d] should be 0.5 (got %.6f)", i,
		                   out[i]);

	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, TEST_TOL_RELAXED,
		                   "uniform softmax grad[%d] should be 0 (got %.6f)", i,
		                   param_grad_item_at(0, i));

	param_clear();
}

/* [B,m,n] = [1,2,2] with two non-uniform rows. Row0 = [0, ln3]: after
   max-subtraction the exps are 1/3 and 1, sum 4/3, normalizing to
   [1/4, 3/4]. Row1 = [ln3, 0] -> [3/4, 1/4]. Confirms the F32 normalize
   produces the right values, not just the trivial uniform case. */
Test(softmax_3d_f32_cov, f32_forward_nonuniform) {
	double l3 = log(3.0);
	double in[4] = {0.0, l3, l3, 0.0};
	int sh[3] = {1, 2, 2};
	TensorHandle x = tensor_create_streamed(hcopy(in, 4), sh, 3, 0, 0, DTAG_F32);

	TensorHandle r = tensor_softmax_3d(x);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "F32 input -> F32 output (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_numel(r), 4);
	double out[4];
	tensor_to_doubles(r, out);
	double expected[4] = {0.25, 0.75, 0.75, 0.25};
	/* exp(ln3) is not exact in single precision, so use an F32-scale tolerance
	   (1e-5) rather than TEST_TOL_RELAXED (1e-10 on tape). */
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected[i], 1e-5, "softmax[%d] should be %.2f (got %.6f)", i,
		                   expected[i], out[i]);
}
#endif

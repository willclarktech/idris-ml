/* Criterion suite `dropout_cov` — coverage top-up for tape nn/norm/dropout.c.
 *
 * The base path leaves the F32 store arms of tensor_dropout uncovered:
 *   - line 38: ((float*)out)[i] = 0.0f;      (F32, element dropped)
 *   - line 45: ((float*)out)[i] = (float)v;  (F32, element survives, scaled)
 * Both sit under `if (is_f32)`, reached only when the input tensor is
 * F32-tagged (DT_F32) AND training=1 with p>0. A single F32 forward with a
 * fixed seed and a length large enough to land elements on both sides of the
 * survival threshold executes both arms.
 *
 * Oracle without predicting the LCG: the inverted-dropout contract pins every
 * output element to exactly one of two values — 0.0 (dropped) or
 * input[i] * scale (survived), scale = 1/(1-p). With p=0.5 the scale is 2 and
 * the inputs are integers, so both candidates are exact in single precision.
 * We assert each element matches one of the two, then assert the run produced
 * at least one of each (so both store arms actually fired — otherwise the test
 * fails loudly rather than silently skipping a line). The mask is recovered
 * through backward: grad[i] = mask[i] = scale for survivors, 0 for dropped,
 * tying the forward store arm to the backward mask arm element-by-element.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

Test(dropout_cov, f32_train_mask_and_scale) {
	param_clear();
	/* n=16 integer inputs; p=0.5 -> scale=2. With a fixed seed the per-element
	   LCG yields a deterministic mix of drops and survivors over 16 draws,
	   exercising both the F32 zero arm (line 38) and the F32 scale arm
	   (line 45). */
	enum { N = 16 };
	double in_src[N];
	for (int i = 0; i < N; i++)
		in_src[i] = (double)(i + 1);
	double p = 0.5;
	double scale = 2.0; /* 1/(1-0.5) */

	TensorHandle in = tensor_create_1d_streamed(N, hcopy(in_src, N), 1, 0, 14);
	param_register("in", in);

	TensorHandle out = tensor_dropout(in, p, /*training=*/1, /*seed=*/12345u);
	cr_assert_eq(tensor_numel(out), N);
	cr_assert_str_eq(tensor_dtype_name(out), "F32", "F32 input -> F32 dropout output (got %s)",
	                 tensor_dtype_name(out));

	double od[N];
	tensor_to_doubles(out, od);

	int n_drop = 0, n_keep = 0;
	for (int i = 0; i < N; i++) {
		double survive = in_src[i] * scale; /* exact integer in F32 */
		int is_zero = (od[i] == 0.0);
		int is_survive = (od[i] == survive);
		cr_assert(is_zero || is_survive, "out[%d]=%.6f must be 0 or input*scale=%.1f", i, od[i],
		          survive);
		if (is_zero)
			n_drop++;
		else
			n_keep++;
	}
	/* Both F32 store arms must have executed for this seed. */
	cr_assert_gt(n_drop, 0, "expected at least one dropped element (line 38 arm)");
	cr_assert_gt(n_keep, 0, "expected at least one survivor (line 45 arm)");

	/* Backward: grad[i] = mask[i] = scale for survivors, 0 for dropped. This
	   pins each forward store arm to the corresponding mask, consistently. */
	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);
	for (int i = 0; i < N; i++) {
		double expected = (od[i] == 0.0) ? 0.0 : scale;
		cr_assert_float_eq(param_grad_item_at(0, i), expected, TEST_TOL_RELAXED,
		                   "grad[%d] should be %.1f (got %.6f)", i, expected,
		                   param_grad_item_at(0, i));
	}

	param_clear();
}

#endif /* BACKEND_TAPE */

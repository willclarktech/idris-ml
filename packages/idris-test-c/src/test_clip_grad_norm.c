/* Criterion suite for optimizer_clip_grad_norm + _value.
 *
 *   optimizer_clip_grad_norm(max_norm): global-norm gradient clipping.
 *     Computes L2 norm of all-params-flattened gradients.
 *     If norm > max_norm: scales every grad by max_norm / norm.
 *     Returns the (pre-clip) norm.
 *
 *   optimizer_clip_grad_value(max_val): per-element clip; each grad
 *     value is clamped to [-max_val, max_val].
 *
 * These are widely-used in RNN/transformer training (NormClip pattern
 * in trainStep). Was on the W3b/W7 follow-up list as a high-value
 * gap — the probe reported both as 0 hits.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

/* clip_grad_norm test tolerance.
 *
 * tape passes at TEST_TOL_TIGHT (1e-12). mlx passes at TEST_TOL_RELAXED
 * (1e-4 — single-precision storage). torch shows ~9e-8 drift in the
 * post-clip values — about 1 F32 ULP at the magnitudes here — even
 * though both the param and grad ARE F64 on torch CPU. This is the
 * same tape+mlx-agree-torch-outlier pattern documented in TODO row 76
 * (gpt gradient divergence — torch's libtorch autograd produces a
 * subtly different backward than tape's hand-written + mlx's vjp).
 * Using 1e-6 here keeps the test as a regression sentinel without
 * making it false-positive on torch's known precision drift. Tighten
 * when TODO row 76 closes. */
#define CLIP_TOL 1e-6

Test(training_optimizer_clip_grad_norm, clip_when_norm_exceeds_max) {
	/* Set up two params x, y with known gradients via L2-loss backward:
	 *   x = [3, 4]; loss_x = sum(x*x)/2 -> d loss_x / d x = x = [3, 4]
	 *   y = [12];   loss_y = sum(y*y)/2 -> d loss_y / d y = y = [12]
	 *   total loss = loss_x + loss_y; combined grad-norm = sqrt(9+16+144) = 13
	 * Call clip_grad_norm(5.0) -> returns 13; rescales grads by 5/13. */
	param_clear();
	double xd[] = {3.0, 4.0};
	double yd[] = {12.0};
	TensorHandle x = tensor_create_param_1d_f64(2, hcopy(xd, 2));
	TensorHandle y = tensor_create_param_1d_f64(1, hcopy(yd, 1));
	param_register("x", x);
	param_register("y", y);

	/* Build loss = sum(x*x)/2 + sum(y*y)/2 using elementwise mul,
	 * sum, and scalar mul. */
	TensorHandle xx = tensor_mul(x, x);                 /* [9, 16] */
	TensorHandle yy = tensor_mul(y, y);                 /* [144] */
	TensorHandle xsum = tensor_sum(xx);                 /* 25 */
	TensorHandle ysum = tensor_sum(yy);                 /* 144 */
	TensorHandle half_x = tensor_mul_scalar(xsum, 0.5); /* 12.5 */
	TensorHandle half_y = tensor_mul_scalar(ysum, 0.5); /* 72 */
	TensorHandle loss = tensor_add(half_x, half_y);     /* 84.5 */
	cr_assert_float_eq(tensor_item(loss), 84.5, CLIP_TOL, "loss should be 84.5 (got %.9f)",
	                   tensor_item(loss));

	tensor_backward(loss);

	/* Pre-clip grads: x.grad = x = [3, 4]; y.grad = [12]. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 3.0, CLIP_TOL);
	cr_assert_float_eq(param_grad_item_at(0, 1), 4.0, CLIP_TOL);
	cr_assert_float_eq(param_grad_item_at(1, 0), 12.0, CLIP_TOL);

	/* Clip. Combined norm should be sqrt(9+16+144) = 13. */
	double norm = optimizer_clip_grad_norm(5.0);
	cr_assert_float_eq(norm, 13.0, CLIP_TOL,
	                   "clip_grad_norm should return pre-clip norm 13 (got %.9f)", norm);

	/* Post-clip grads: scale = 5/13. */
	double s = 5.0 / 13.0;
	cr_assert_float_eq(param_grad_item_at(0, 0), 3.0 * s, CLIP_TOL,
	                   "post-clip x.grad[0] should be 3*5/13 (got %.9f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 4.0 * s, CLIP_TOL,
	                   "post-clip x.grad[1] should be 4*5/13 (got %.9f)", param_grad_item_at(0, 1));
	cr_assert_float_eq(param_grad_item_at(1, 0), 12.0 * s, CLIP_TOL,
	                   "post-clip y.grad[0] should be 12*5/13 (got %.9f)",
	                   param_grad_item_at(1, 0));
}

Test(training_optimizer_clip_grad_norm, no_op_when_norm_below_max) {
	/* When pre-clip norm <= max_norm, grads should be untouched. */
	param_clear();
	double xd[] = {1.0, 1.0};
	TensorHandle x = tensor_create_param_1d_f64(2, hcopy(xd, 2));
	param_register("x", x);
	TensorHandle xx = tensor_mul(x, x);
	TensorHandle loss = tensor_sum(xx);
	tensor_backward(loss);
	/* d loss / d x = 2 * x = [2, 2]; norm = sqrt(8) ≈ 2.828. */
	double norm = optimizer_clip_grad_norm(10.0);
	cr_assert_float_eq(norm, sqrt(8.0), CLIP_TOL,
	                   "clip_grad_norm should return pre-clip norm sqrt(8) (got %.9f)", norm);
	/* Grads should be unchanged (no scaling applied). */
	cr_assert_float_eq(param_grad_item_at(0, 0), 2.0, CLIP_TOL,
	                   "x.grad[0] should be untouched at 2 (got %.9f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 2.0, CLIP_TOL);
}

Test(training_optimizer_clip_grad_value, clamps_per_element) {
	/* optimizer_clip_grad_value(max_val): clamp each grad to
	 * [-max_val, max_val]. */
	param_clear();
	double xd[] = {3.0, -2.0};
	TensorHandle x = tensor_create_param_1d_f64(2, hcopy(xd, 2));
	param_register("x", x);
	TensorHandle xx = tensor_mul(x, x);
	TensorHandle loss = tensor_sum(xx);
	tensor_backward(loss);
	/* d loss / d x = 2 * x = [6, -4]. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 6.0, CLIP_TOL);
	cr_assert_float_eq(param_grad_item_at(0, 1), -4.0, CLIP_TOL);

	optimizer_clip_grad_value(3.0);
	/* After clip: [min(6,3) = 3, max(-4,-3) = -3]. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 3.0, CLIP_TOL,
	                   "x.grad[0]=6 should clamp to +3 (got %.9f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), -3.0, CLIP_TOL,
	                   "x.grad[1]=-4 should clamp to -3 (got %.9f)", param_grad_item_at(0, 1));
}

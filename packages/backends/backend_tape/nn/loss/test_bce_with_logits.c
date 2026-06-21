/* Criterion suite for `tensor_bce_with_logits` (BCE-with-logits loss).
 *
 * Forward: mean over i of max(p,0) - p*y + log(1+exp(-|p|)).
 * Backward (OP_BCE_WITH_LOGITS): d_input[i] = (sigmoid(p_i) - y_i) / n.
 *
 * The op's backward was registered but never tested — the probe reported
 * OP_BCE_WITH_LOGITS "present" via impl self-match while the file sat at
 * ~0% line coverage.
 *
 * RED before this commit: the backward assertion grad == (sigmoid(p)-y)/n
 * (e.g. -0.5 for p=0,y=1,n=1) fails if the backward is unrun or wrong;
 * the forward assertion log(2) for p=0,y=0 likewise pins the formula.
 */

#include <math.h>
#include <criterion/criterion.h>
#include "test_helpers.h"

Test(nn_loss_bce_with_logits, forward_zero_logits_zero_target) {
	/* term = max(0,0) - 0 + log(1+exp(0)) = log 2 per element; mean = log 2 */
	double pd[] = {0.0, 0.0};
	double yd[] = {0.0, 0.0};
	int s[] = {2};
	TensorHandle p = tensor_create(pd, s, 1, 0);
	TensorHandle y = tensor_create(yd, s, 1, 0);
	cr_assert_float_eq(tensor_item(tensor_bce_with_logits(p, y)), log(2.0), TEST_TOL_RELAXED);
}

Test(nn_loss_bce_with_logits, forward_known_value) {
	/* p=2, y=1, n=1: 2 - 2 + log(1+exp(-2)) = log(1+exp(-2)) */
	double pd[] = {2.0};
	double yd[] = {1.0};
	int s[] = {1};
	TensorHandle p = tensor_create(pd, s, 1, 0);
	TensorHandle y = tensor_create(yd, s, 1, 0);
	double expected = log(1.0 + exp(-2.0));
	cr_assert_float_eq(tensor_item(tensor_bce_with_logits(p, y)), expected, TEST_TOL_RELAXED);
}

/* Backward derivative is the closed form sigmoid(p)-y. The loss has a
 * kink at p==0 (relu(p) and |p|): all three backends now agree there.
 * tape computes the closed form directly; torch uses libtorch's fused
 * backward; mlx records one OP_BCE_WITH_LOGITS entry whose replay uses
 * the smooth softplus form (log(1+exp(p))), so its vjp gives sigmoid(p)
 * at the kink instead of the subgradient the decomposed relu/|p| path
 * picked. The backward_at_kink test below pins p==0 across all three. */

Test(nn_loss_bce_with_logits, backward_scalar) {
	/* p=1, y=1, n=1: d_input = sigmoid(1) - 1 */
	param_clear();
	double pd[] = {1.0};
	double yd[] = {1.0};
	int s[] = {1};
	TensorHandle p = tensor_create(pd, s, 1, 1);
	TensorHandle y = tensor_create(yd, s, 1, 0);
	param_register("p", p);
	TensorHandle loss = tensor_bce_with_logits(p, y);
	tensor_backward(loss);
	double expected = 1.0 / (1.0 + exp(-1.0)) - 1.0; /* sigmoid(1) - 1 */
	cr_assert_float_eq(param_grad_item_at(0, 0), expected, TEST_TOL_RELAXED,
	                   "d_bce/dp = sigmoid(1)-1 (got %.6f exp %.6f)", param_grad_item_at(0, 0),
	                   expected);
}

Test(nn_loss_bce_with_logits, backward_at_kink) {
	/* p=0, y=1, n=1: the kink. Closed form d_input = sigmoid(0) - 1 = -0.5.
	   Pre-fix, mlx's decomposed relu/|p| replay returned the subgradient
	   -1.0 here; tape/torch already gave -0.5. All three must now agree. */
	param_clear();
	double pd[] = {0.0};
	double yd[] = {1.0};
	int s[] = {1};
	TensorHandle p = tensor_create(pd, s, 1, 1);
	TensorHandle y = tensor_create(yd, s, 1, 0);
	param_register("p", p);
	TensorHandle loss = tensor_bce_with_logits(p, y);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), -0.5, TEST_TOL_RELAXED,
	                   "d_bce/dp at kink p=0 = sigmoid(0)-1 = -0.5 (got %.6f)",
	                   param_grad_item_at(0, 0));
}

Test(nn_loss_bce_with_logits, backward_vector_mean_scaled) {
	/* p=[1,-1], y=[1,0], n=2: grad = [(sigmoid(1)-1)/2, (sigmoid(-1)-0)/2].
	   The /n mean-scaling is the property this pins. */
	param_clear();
	double pd[] = {1.0, -1.0};
	double yd[] = {1.0, 0.0};
	int s[] = {2};
	TensorHandle p = tensor_create(pd, s, 1, 1);
	TensorHandle y = tensor_create(yd, s, 1, 0);
	param_register("p", p);
	TensorHandle loss = tensor_bce_with_logits(p, y);
	tensor_backward(loss);
	double sig1 = 1.0 / (1.0 + exp(-1.0));
	double sigm1 = 1.0 / (1.0 + exp(1.0));
	cr_assert_float_eq(param_grad_item_at(0, 0), (sig1 - 1.0) / 2.0, TEST_TOL_RELAXED, "grad[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), (sigm1 - 0.0) / 2.0, TEST_TOL_RELAXED, "grad[1]");
}

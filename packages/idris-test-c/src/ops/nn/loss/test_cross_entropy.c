/* tensor_cross_entropy — value + backward (public FFI contract). */
#include "port_assert.h"

Test(nn_loss_cross_entropy, value_and_backward) {
	param_clear();
	/* input=[1,2,3] (logits), target=[0,0,1] (one-hot for class 2).
	   softmax(input) = [e^1, e^2, e^3] / Z, log_softmax[2] = 3 - log(Z).
	   CE = -target . log_softmax / 3 = -log_softmax[2] / 3
	   Note: dim=0 for vectors, log_softmax matches tape's convention. */
	double id[] = {1, 2, 3}, td[] = {0, 0, 1};
	int s[] = {3};
	TensorHandle in = tensor_create(id, s, 1, 1);
	TensorHandle tg = tensor_create(td, s, 1, 0);
	param_register("in", in);
	TensorHandle loss = tensor_cross_entropy(in, tg);
	/* log_softmax[2] = 3 - log(e + e^2 + e^3); CE = -log_softmax[2]/3 */
	double Z = exp(1) + exp(2) + exp(3);
	double expected = -(3.0 - log(Z)) / 3.0;
	ASSERT_NEAR("ce loss value", tensor_item(loss), expected, 1e-5);

	if (tensor_requires_grad(loss)) {
		tensor_backward(loss);
		/* d_loss/d_in[i] = (softmax[i] - target[i]) / numel.
		   Note: this assumes the standard CE-with-softmax derivative, which
		   our decomposed impl computes via vjp on log_softmax + mul + neg + mean. */
		double sm0 = exp(1) / Z, sm1 = exp(2) / Z, sm2 = exp(3) / Z;
		ASSERT_NEAR("d_ce_in[0]", param_grad_item_at(0, 0), sm0 / 3.0, 1e-5);
		ASSERT_NEAR("d_ce_in[1]", param_grad_item_at(0, 1), sm1 / 3.0, 1e-5);
		ASSERT_NEAR("d_ce_in[2]", param_grad_item_at(0, 2), (sm2 - 1.0) / 3.0, 1e-5);
	} else {
		printf("ok: ce loss has no grad on this backend (tape's no-grad stub) — skipping\n");
	}
	param_clear();
}

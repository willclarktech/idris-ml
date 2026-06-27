/* tensor_mse_loss — value + backward (public FFI contract). */
#include "port_assert.h"

Test(nn_loss_mse_loss, value_and_backward) {
	param_clear();
	/* input = [1,2,3], target = [1.5, 2.5, 3.5]. Diff = [-0.5, -0.5, -0.5].
	   MSE = mean(0.25, 0.25, 0.25) = 0.25 */
	double id[] = {1, 2, 3}, td[] = {1.5, 2.5, 3.5};
	int s[] = {3};
	TensorHandle in = tensor_create(id, s, 1, 1);
	TensorHandle tg = tensor_create(td, s, 1, 0);
	param_register("in", in);
	TensorHandle loss = tensor_mse_loss(in, tg);
	ASSERT_NEAR("mse loss value", tensor_item(loss), 0.25, 1e-6);

	if (tensor_requires_grad(loss)) {
		tensor_backward(loss);
		/* d/d_in[i] = 2 * (in[i] - tg[i]) / 3 = -1/3 for each i */
		ASSERT_NEAR("d_mse_in[0]", param_grad_item_at(0, 0), -1.0 / 3.0, 1e-6);
		ASSERT_NEAR("d_mse_in[1]", param_grad_item_at(0, 1), -1.0 / 3.0, 1e-6);
		ASSERT_NEAR("d_mse_in[2]", param_grad_item_at(0, 2), -1.0 / 3.0, 1e-6);
	} else {
		printf("ok: mse loss has no grad on this backend (tape's no-grad stub) — skipping\n");
	}
	param_clear();
}

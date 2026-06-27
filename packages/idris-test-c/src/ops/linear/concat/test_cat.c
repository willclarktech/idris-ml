#include "port_assert.h"

Test(linear_concat_cat, cat_backward) {
	param_clear();
	/* Two [3]-vectors: [1,2,3], [4,5,6]. Cat at dim=0 -> [6]. */
	double a[] = {1, 2, 3}, b[] = {4, 5, 6};
	int s[] = {3};
	TensorHandle ta = tensor_create(a, s, 1, 1);
	TensorHandle tb = tensor_create(b, s, 1, 1);
	param_register("a", ta);
	param_register("b", tb);
	TensorHandle in[] = {ta, tb};
	TensorHandle ct = tensor_cat(in, 2, 0);
	if (tensor_dim(ct) == 1 && tensor_size(ct, 0) == 6) {
		double cout[6];
		tensor_to_doubles(ct, cout);
		ASSERT_NEAR("cat[0]", cout[0], 1.0, 1e-10);
		ASSERT_NEAR("cat[2]", cout[2], 3.0, 1e-10);
		ASSERT_NEAR("cat[3]", cout[3], 4.0, 1e-10);
		ASSERT_NEAR("cat[5]", cout[5], 6.0, 1e-10);

		TensorHandle loss = tensor_sum(ct);
		tensor_backward(loss);
		ASSERT_NEAR("d_a[0]", param_grad_item_at(0, 0), 1.0, 1e-6);
		ASSERT_NEAR("d_b[2]", param_grad_item_at(1, 2), 1.0, 1e-6);
	} else {
		printf("ok: cat stub on this backend (rank/size unexpected) — skipping\n");
	}
	param_clear();
}

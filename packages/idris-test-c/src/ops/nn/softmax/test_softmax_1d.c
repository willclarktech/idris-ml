#include "port_assert.h"

Test(nn_softmax_softmax_1d, softmax) {
	double data[] = {1.0, 2.0, 3.0};
	int shape[] = {3};
	TensorHandle v = tensor_create(data, shape, 1, 0);
	TensorHandle sm = tensor_softmax(v, 0);
	TensorHandle s = tensor_sum(sm);
	ASSERT_NEAR("softmax sums to 1", tensor_item(s), 1.0, 1e-6);
	tensor_free(v);
	tensor_free(sm);
	tensor_free(s);
}

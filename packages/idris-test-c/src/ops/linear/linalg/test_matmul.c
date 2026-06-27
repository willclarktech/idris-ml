#include "port_assert.h"

Test(linear_linalg_matmul, mm_backward) {
	param_clear();

	double a_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
	double b_data[] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
	int a_shape[] = {2, 3};
	int b_shape[] = {3, 2};

	/* Analytical gradient */
	TensorHandle a = tensor_create(a_data, a_shape, 2, 1);
	param_register("a", a);
	TensorHandle b = tensor_create(b_data, b_shape, 2, 1);
	param_register("b", b);

	TensorHandle c = tensor_mm(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);

	/* Capture analytical grads BEFORE param_clear — mlx's param_clear
	   actually releases the registry (correct per refcount lifecycle),
	   so post-clear reads on mlx see an empty registry. Tape's
	   param_clear is count-only and accidentally tolerates the pattern. */
	double analytic_a00 = param_grad_item_at(0, 0);

	/* Finite diff check for a[0,0] */
	double eps = 1e-5;
	double a_copy[6];
	memcpy(a_copy, a_data, 6 * sizeof(double));
	a_copy[0] += eps;
	{
		param_clear();
		TensorHandle a2 = tensor_create(a_copy, a_shape, 2, 0);
		TensorHandle b2 = tensor_create(b_data, b_shape, 2, 0);
		double f_plus = tensor_item(tensor_sum(tensor_mm(a2, b2)));
		a_copy[0] = a_data[0] - eps;
		TensorHandle a3 = tensor_create(a_copy, a_shape, 2, 0);
		TensorHandle b3 = tensor_create(b_data, b_shape, 2, 0);
		double f_minus = tensor_item(tensor_sum(tensor_mm(a3, b3)));
		double fd = (f_plus - f_minus) / (2 * eps);
		printf("  a[0,0]: fd=%f analytic=%f err=%e\n", fd, analytic_a00, fabs(fd - analytic_a00));
		ASSERT_NEAR("mm grad a[0,0]", analytic_a00, fd, FD_TOL);
	}

	param_clear();
}

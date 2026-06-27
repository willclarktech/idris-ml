#include "port_assert.h"

Test(linear_linalg_linear_2d, linear_2d_forward) {
	/* W: [2, 3] (o=2, i=3), X: [4, 3] (B=4), bias: [2] */
	double w_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
	double x_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5, 1.5, 2.5};
	double b_data[] = {10.0, 20.0};
	int w_shape[] = {2, 3};
	int x_shape[] = {4, 3};
	int b_shape[] = {2};

	TensorHandle W = tensor_create(w_data, w_shape, 2, 0);
	TensorHandle X = tensor_create(x_data, x_shape, 2, 0);
	TensorHandle bias = tensor_create(b_data, b_shape, 1, 0);

	TensorHandle Y = tensor_linear_2d(W, X, bias);

	/* Y[b,o] = sum_i X[b,i] * W[o,i] + bias[o]
	   Y[0,0] = 1*0.1 + 2*0.2 + 3*0.3 + 10 = 0.1+0.4+0.9+10 = 11.4
	   Y[0,1] = 1*0.4 + 2*0.5 + 3*0.6 + 20 = 0.4+1.0+1.8+20 = 23.2
	   Y[1,0] = 4*0.1 + 5*0.2 + 6*0.3 + 10 = 0.4+1.0+1.8+10 = 13.2
	   Y[1,1] = 4*0.4 + 5*0.5 + 6*0.6 + 20 = 1.6+2.5+3.6+20 = 27.7 */
	ASSERT_NEAR("lin2d Y[0,0]", tensor_item_2d(Y, 0, 0), 11.4, VAL_TOL);
	ASSERT_NEAR("lin2d Y[0,1]", tensor_item_2d(Y, 0, 1), 23.2, VAL_TOL);
	ASSERT_NEAR("lin2d Y[1,0]", tensor_item_2d(Y, 1, 0), 13.2, VAL_TOL);
	ASSERT_NEAR("lin2d Y[1,1]", tensor_item_2d(Y, 1, 1), 27.7, VAL_TOL);

	/* B=4 row: Y[3,0] = 0.5*0.1+1.5*0.2+2.5*0.3 + 10 = 0.05+0.3+0.75+10 = 11.1 */
	ASSERT_NEAR("lin2d Y[3,0]", tensor_item_2d(Y, 3, 0), 11.1, VAL_TOL);
}

Test(linear_linalg_linear_2d, linear_2d_matches_per_sample) {
	/* For B independent inputs, batched tensor_linear_2d must produce the
	   same outputs as B calls to tensor_linear (per-sample mv+bias). */
	double w_data[] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
	double b_data[] = {0.7, -0.8};
	int w_shape[] = {2, 3};
	int b_shape[] = {2};

	/* Three inputs */
	double x0_data[] = {1.0, 2.0, 3.0};
	double x1_data[] = {-1.0, 0.5, 0.25};
	double x2_data[] = {0.0, -2.0, 1.5};
	int x_shape[] = {3};

	TensorHandle W = tensor_create(w_data, w_shape, 2, 0);
	TensorHandle bias = tensor_create(b_data, b_shape, 1, 0);

	/* Per-sample */
	TensorHandle x0 = tensor_create(x0_data, x_shape, 1, 0);
	TensorHandle x1 = tensor_create(x1_data, x_shape, 1, 0);
	TensorHandle x2 = tensor_create(x2_data, x_shape, 1, 0);
	TensorHandle y0 = tensor_linear(W, x0, bias);
	TensorHandle y1 = tensor_linear(W, x1, bias);
	TensorHandle y2 = tensor_linear(W, x2, bias);

	/* Batched */
	double xb_data[] = {1.0, 2.0, 3.0, -1.0, 0.5, 0.25, 0.0, -2.0, 1.5};
	int xb_shape[] = {3, 3};
	TensorHandle Xb = tensor_create(xb_data, xb_shape, 2, 0);
	TensorHandle Yb = tensor_linear_2d(W, Xb, bias);

	ASSERT_NEAR("Yb[0,0]==y0[0]", tensor_item_2d(Yb, 0, 0), tensor_item_1d(y0, 0), 1e-9);
	ASSERT_NEAR("Yb[0,1]==y0[1]", tensor_item_2d(Yb, 0, 1), tensor_item_1d(y0, 1), 1e-9);
	ASSERT_NEAR("Yb[1,0]==y1[0]", tensor_item_2d(Yb, 1, 0), tensor_item_1d(y1, 0), 1e-9);
	ASSERT_NEAR("Yb[1,1]==y1[1]", tensor_item_2d(Yb, 1, 1), tensor_item_1d(y1, 1), 1e-9);
	ASSERT_NEAR("Yb[2,0]==y2[0]", tensor_item_2d(Yb, 2, 0), tensor_item_1d(y2, 0), 1e-9);
	ASSERT_NEAR("Yb[2,1]==y2[1]", tensor_item_2d(Yb, 2, 1), tensor_item_1d(y2, 1), 1e-9);
}

Test(linear_linalg_linear_2d, linear_2d_backward) {
	param_clear();

	double w_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
	double x_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double b_data[] = {0.7, 0.8};
	int w_shape[] = {2, 3};
	int x_shape[] = {2, 3};
	int b_shape[] = {2};

	TensorHandle W = tensor_create(w_data, w_shape, 2, 1);
	param_register("W", W);
	TensorHandle X = tensor_create(x_data, x_shape, 2, 1);
	param_register("X", X);
	TensorHandle bias = tensor_create(b_data, b_shape, 1, 1);
	param_register("bias", bias);

	TensorHandle Y = tensor_linear_2d(W, X, bias);
	TensorHandle loss = tensor_sum(Y);
	tensor_backward(loss);

	/* Analytical:
	   dY/dW[o,i] = sum_b X[b,i]   (since loss = sum_b sum_o Y[b,o])
	   dY/dX[b,i] = sum_o W[o,i]
	   dY/dbias[o] = B (number of batch elements) */

	/* W[0,0]: sum_b X[b,0] = 1 + 4 = 5 */
	{
		double eps = 1e-5;
		double w_copy[6];
		memcpy(w_copy, w_data, 6 * sizeof(double));
		w_copy[0] = w_data[0] + eps;
		TensorHandle Wp = tensor_create(w_copy, w_shape, 2, 0);
		TensorHandle Xp = tensor_create(x_data, x_shape, 2, 0);
		TensorHandle Bp = tensor_create(b_data, b_shape, 1, 0);
		double f_plus = tensor_item(tensor_sum(tensor_linear_2d(Wp, Xp, Bp)));
		w_copy[0] = w_data[0] - eps;
		TensorHandle Wm = tensor_create(w_copy, w_shape, 2, 0);
		TensorHandle Xm = tensor_create(x_data, x_shape, 2, 0);
		TensorHandle Bm = tensor_create(b_data, b_shape, 1, 0);
		double f_minus = tensor_item(tensor_sum(tensor_linear_2d(Wm, Xm, Bm)));
		double fd = (f_plus - f_minus) / (2 * eps);
		double analytic = param_grad_item_at(0, 0);
		printf("  W[0,0]: fd=%f analytic=%f\n", fd, analytic);
		ASSERT_NEAR("lin2d grad W[0,0]", analytic, fd, FD_TOL);
	}

	/* X[0,0]: sum_o W[o,0] = 0.1 + 0.4 = 0.5 */
	{
		double eps = 1e-5;
		double x_copy[6];
		memcpy(x_copy, x_data, 6 * sizeof(double));
		x_copy[0] = x_data[0] + eps;
		TensorHandle Wp = tensor_create(w_data, w_shape, 2, 0);
		TensorHandle Xp = tensor_create(x_copy, x_shape, 2, 0);
		TensorHandle Bp = tensor_create(b_data, b_shape, 1, 0);
		double f_plus = tensor_item(tensor_sum(tensor_linear_2d(Wp, Xp, Bp)));
		x_copy[0] = x_data[0] - eps;
		TensorHandle Wm = tensor_create(w_data, w_shape, 2, 0);
		TensorHandle Xm = tensor_create(x_copy, x_shape, 2, 0);
		TensorHandle Bm = tensor_create(b_data, b_shape, 1, 0);
		double f_minus = tensor_item(tensor_sum(tensor_linear_2d(Wm, Xm, Bm)));
		double fd = (f_plus - f_minus) / (2 * eps);
		double analytic = param_grad_item_at(1, 0);
		printf("  X[0,0]: fd=%f analytic=%f\n", fd, analytic);
		ASSERT_NEAR("lin2d grad X[0,0]", analytic, fd, FD_TOL);
	}

	/* bias[0]: B = 2 */
	{
		double analytic = param_grad_item_at(2, 0);
		printf("  bias[0]: analytic=%f (expected 2.0)\n", analytic);
		ASSERT_NEAR("lin2d grad bias[0]", analytic, 2.0, 1e-9);
	}

	param_clear();
}

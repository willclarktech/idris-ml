/* Criterion suite for tensor_batch_norm running-statistics update.
 *
 *   batch_norm(input, gamma, beta, running_mean, running_var,
 *              C, spatial, training, momentum, eps):
 *     training mode normalizes per channel over the spatial dim using the
 *     *biased* batch variance, and blends running_mean/running_var via the
 *     momentum EMA.
 *
 * Cross-backend invariant (PyTorch is the oracle, feedback_paired_side_alignment):
 * the running_var update applies Bessel's n/(n-1) correction to the batch
 * variance — torch delegates to torch::batch_norm which does this; tape + mlx
 * must match. The per-batch normalization stays biased (not asserted here).
 *
 * Lives in backend_tape/ but the test glob compiles it into every backend's
 * test binary, so the same oracle is asserted on tape / torch / mlx.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"
#include "port_assert.h"

Test(nn_norm_batch_norm, running_var_bessel_corrected) {
	/* C=1, spatial=4, input = [1, 2, 3, 4], momentum=0.1.
	 * mean = 2.5, biased batch var = 1.25.
	 * running_mean = 0.9*0 + 0.1*2.5            = 0.25
	 * running_var  = 0.9*1 + 0.1*1.25*(4/3)     = 1.0666667  (Bessel n/(n-1))
	 * The buggy biased path yields 0.9 + 0.1*1.25 = 1.025.
	 */
	param_clear();
	double in_d[] = {1.0, 2.0, 3.0, 4.0};
	double g_d[] = {1.0};
	double b_d[] = {0.0};
	double rm_d[] = {0.0};
	double rv_d[] = {1.0};
	TensorHandle input = tensor_create_2d_f64(1, 4, hcopy(in_d, 4), 0);
	TensorHandle gamma = tensor_create_1d_f64(1, hcopy(g_d, 1), 0);
	TensorHandle beta = tensor_create_1d_f64(1, hcopy(b_d, 1), 0);
	TensorHandle running_mean = tensor_create_1d_f64(1, hcopy(rm_d, 1), 0);
	TensorHandle running_var = tensor_create_1d_f64(1, hcopy(rv_d, 1), 0);
	double momentum = 0.1;
	double eps = 1e-5;
	tensor_batch_norm(input, gamma, beta, running_mean, running_var, 1, 4, 1, momentum, eps);

	double rv_out[1];
	double rm_out[1];
	tensor_to_doubles(running_var, rv_out);
	tensor_to_doubles(running_mean, rm_out);

	double expected_rv = 0.9 * 1.0 + 0.1 * 1.25 * (4.0 / 3.0);
	cr_assert_float_eq(rv_out[0], expected_rv, TEST_TOL_RELAXED,
	                   "running_var should be Bessel-corrected %.9f (got %.9f)", expected_rv,
	                   rv_out[0]);
	double expected_rm = 0.9 * 0.0 + 0.1 * 2.5;
	cr_assert_float_eq(rm_out[0], expected_rm, TEST_TOL_RELAXED,
	                   "running_mean should be %.9f (got %.9f)", expected_rm, rm_out[0]);
}

Test(nn_norm_batch_norm, batch_norm_forward) {
	/* Input: [2 channels, 3 spatial] = flat [6] */
	double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {6};
	TensorHandle inp = tensor_create(data, shape, 1, 0);

	double gamma_d[] = {1.0, 1.0};
	double beta_d[] = {0.0, 0.0};
	double rm_d[] = {0.0, 0.0};
	double rv_d[] = {1.0, 1.0};
	int s1[] = {2};
	TensorHandle gamma = tensor_create(gamma_d, s1, 1, 0);
	TensorHandle beta = tensor_create(beta_d, s1, 1, 0);
	TensorHandle rm = tensor_create(rm_d, s1, 1, 0);
	TensorHandle rv = tensor_create(rv_d, s1, 1, 0);

	/* Training mode: normalize using input stats */
	TensorHandle out = tensor_batch_norm(inp, gamma, beta, rm, rv, 2, 3, 1, 0.1, 1e-5);

	/* Channel 0: mean=2, var=2/3, x_hat = [-1.22, 0, 1.22] (approx) */
	double result[6];
	tensor_to_doubles(out, result);
	ASSERT_NEAR("bn ch0 mean~0", (result[0] + result[1] + result[2]) / 3.0, 0.0, 1e-4);
	ASSERT_NEAR("bn ch1 mean~0", (result[3] + result[4] + result[5]) / 3.0, 0.0, 1e-4);

	/* Eval mode: should use running stats */
	TensorHandle out2 = tensor_batch_norm(inp, gamma, beta, rm, rv, 2, 3, 0, 0.1, 1e-5);
	double result2[6];
	tensor_to_doubles(out2, result2);
	/* Running mean was updated — eval output should differ from training output */
	printf("ok: batch norm forward runs\n");
}

Test(nn_norm_batch_norm, batch_norm_backward) {
	param_clear();

	double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {6};
	TensorHandle inp = tensor_create(data, shape, 1, 1);
	param_register("inp", inp);

	double gamma_d[] = {1.0, 1.0};
	double beta_d[] = {0.0, 0.0};
	double rm_d[] = {0.0, 0.0};
	double rv_d[] = {1.0, 1.0};
	int s1[] = {2};
	double* g_buf = hcopy(gamma_d, 2);
	TensorHandle gamma = tensor_create_param_1d_f64(2, g_buf);
	double* b_buf = hcopy(beta_d, 2);
	TensorHandle beta = tensor_create_param_1d_f64(2, b_buf);
	TensorHandle rm = tensor_create(rm_d, s1, 1, 0);
	TensorHandle rv = tensor_create(rv_d, s1, 1, 0);

	TensorHandle out = tensor_batch_norm(inp, gamma, beta, rm, rv, 2, 3, 1, 0.1, 1e-5);
	TensorHandle loss = tensor_sum(out);
	tensor_backward(loss);

	/* d_beta[c] = sum of output grads for that channel = 3 * 1.0 = 3.0 */
	/* But output is normalized, so d_beta[c] = sum(1.0) = 3.0 for each channel */
	/* d_gamma: sum of x_hat * grad. Since mean(x_hat)=0, sum(x_hat)=0 → d_gamma=0 */

	/* Finite diff check: perturb gamma[0] */
	double eps = 1e-5;
	{
		param_clear();
		double gp[] = {1.0 + eps, 1.0};
		double gm[] = {1.0 - eps, 1.0};
		double* gp_buf = hcopy(gp, 2);
		double* gm_buf = hcopy(gm, 2);
		double* b1 = hcopy(beta_d, 2);
		double* b2 = hcopy(beta_d, 2);

		TensorHandle i1 = tensor_create(data, shape, 1, 0);
		TensorHandle g1 = tensor_create(gp, s1, 1, 0);
		TensorHandle bt1 = tensor_create(beta_d, s1, 1, 0);
		TensorHandle rm1 = tensor_create(rm_d, s1, 1, 0);
		TensorHandle rv1 = tensor_create(rv_d, s1, 1, 0);
		double fp =
		    tensor_item(tensor_sum(tensor_batch_norm(i1, g1, bt1, rm1, rv1, 2, 3, 1, 0.1, 1e-5)));

		TensorHandle i2 = tensor_create(data, shape, 1, 0);
		TensorHandle g2 = tensor_create(gm, s1, 1, 0);
		TensorHandle bt2 = tensor_create(beta_d, s1, 1, 0);
		TensorHandle rm2 = tensor_create(rm_d, s1, 1, 0);
		TensorHandle rv2 = tensor_create(rv_d, s1, 1, 0);
		double fm =
		    tensor_item(tensor_sum(tensor_batch_norm(i2, g2, bt2, rm2, rv2, 2, 3, 1, 0.1, 1e-5)));

		double fd = (fp - fm) / (2 * eps);
		/* d_gamma[0] should be ~0 (sum of x_hat for centered data) */
		ASSERT_NEAR("bn fd d_gamma[0]", fd, 0.0, 0.2);
		(void)gp_buf;
		(void)gm_buf;
		(void)b1;
		(void)b2;
	}
	param_clear();
}

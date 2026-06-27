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

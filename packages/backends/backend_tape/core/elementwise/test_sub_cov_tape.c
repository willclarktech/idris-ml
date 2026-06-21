/* Criterion suite `sub_cov` — coverage top-up for tape core/elementwise/sub.c.
 *
 * The base suite covers the F64 same-shape and scalar-broadcast backward arms
 * of tape_backward_sub. This file closes the uncovered *general-broadcast*
 * stride arms (sub.c:66-77): the per-element index walk that scatters the
 * upstream grad through compute_bcast_strides when an operand is neither
 * shape-matched nor a scalar.
 *
 *   - do_a arm (sub.c:67-69): left operand broadcasts (a=[3] vs r=[2,3]).
 *   - do_b arm (sub.c:73-76): right operand broadcasts (b=[3] vs r=[2,3]).
 *
 * These are F64 arms (tape_backward_sub accumulates through tape_grad_*_d,
 * double-typed regardless of dtype), so inputs use plain tensor_create (which
 * copies — stack arrays are safe) and grads read at TEST_TOL_TIGHT. Oracles
 * are computed by hand. Wrapped in BACKEND_TAPE: the file lives in a tape dir
 * but is compiled into every backend's test binary, and the broadcast-backward
 * scatter being driven here is the tape implementation specifically.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* a=[3] broadcasts over r=[2,3]; b=[2,3] matches r.
   a=[10,20,30] -> [[10,20,30],[10,20,30]], b=[[1,2,3],[4,5,6]].
   r = a - b = [[9,18,27],[6,15,24]]. loss = sum, dr all 1.
   da[j] = sum over batch of 1 = 2 (scattered via a_str=[0,1]); db[k] = -1.
   Drives the do_a general-broadcast stride arm (sub.c:67-69). */
Test(sub_cov, bcast_backward_left_operand) {
	param_clear();
	double ad[] = {10.0, 20.0, 30.0};
	double bd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int sa[] = {3};
	int sb[] = {2, 3};
	TensorHandle a = tensor_create(ad, sa, 1, 1);
	TensorHandle b = tensor_create(bd, sb, 2, 1);
	param_register("a", a);
	param_register("b", b);

	TensorHandle r = tensor_sub(a, b);
	cr_assert_eq(tensor_numel(r), 6);
	double out[6];
	tensor_to_doubles(r, out);
	double expected_r[] = {9, 18, 27, 6, 15, 24};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expected_r[i], TEST_TOL_TIGHT, "r[%d] should be %.1f (got %.9f)",
		                   i, expected_r[i], out[i]);

	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);

	/* da: left operand broadcasts over batch dim -> each entry summed twice. */
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(0, j), 2.0, TEST_TOL_TIGHT,
		                   "da[%d] should be 2 (got %.9f)", j, param_grad_item_at(0, j));
	/* db: shape-matched, sign-flipped -> -1 everywhere. */
	for (int k = 0; k < 6; k++)
		cr_assert_float_eq(param_grad_item_at(1, k), -1.0, TEST_TOL_TIGHT,
		                   "db[%d] should be -1 (got %.9f)", k, param_grad_item_at(1, k));

	param_clear();
}

/* a=[2,3] matches r=[2,3]; b=[3] broadcasts over r.
   a=[[1,2,3],[4,5,6]], b=[10,20,30] -> [[10,20,30],[10,20,30]].
   r = a - b = [[-9,-18,-27],[-6,-15,-24]]. loss = sum, dr all 1.
   da[k] = 1; db[j] = -(sum over batch of 1) = -2 (scattered via b_str=[0,1]).
   Drives the do_b general-broadcast stride arm (sub.c:73-76). */
Test(sub_cov, bcast_backward_right_operand) {
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	double bd[] = {10.0, 20.0, 30.0};
	int sa[] = {2, 3};
	int sb[] = {3};
	TensorHandle a = tensor_create(ad, sa, 2, 1);
	TensorHandle b = tensor_create(bd, sb, 1, 1);
	param_register("a", a);
	param_register("b", b);

	TensorHandle r = tensor_sub(a, b);
	cr_assert_eq(tensor_numel(r), 6);
	double out[6];
	tensor_to_doubles(r, out);
	double expected_r[] = {-9, -18, -27, -6, -15, -24};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expected_r[i], TEST_TOL_TIGHT, "r[%d] should be %.1f (got %.9f)",
		                   i, expected_r[i], out[i]);

	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);

	/* da: shape-matched -> 1 everywhere. */
	for (int k = 0; k < 6; k++)
		cr_assert_float_eq(param_grad_item_at(0, k), 1.0, TEST_TOL_TIGHT,
		                   "da[%d] should be 1 (got %.9f)", k, param_grad_item_at(0, k));
	/* db: right operand broadcasts over batch dim, sign-flipped -> -2 each. */
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(1, j), -2.0, TEST_TOL_TIGHT,
		                   "db[%d] should be -2 (got %.9f)", j, param_grad_item_at(1, j));

	param_clear();
}

#endif /* BACKEND_TAPE */

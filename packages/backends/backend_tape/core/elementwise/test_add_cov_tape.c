/* Criterion suite `add_cov` — coverage top-up for tape core/elementwise/add.c.
 *
 * The pre-existing add tests cover same-shape and scalar-operand backward
 * (the fast loop and the numel==1 sum-reduce arms). They never reach the
 * general numpy-style broadcast backward block (add.c:70-98), where a
 * non-scalar operand whose shape differs from the result is summed over its
 * broadcast axes via per-position broadcast strides.
 *
 * In particular the `do_a` arm (add.c:79-84, including the uncovered
 * stride-accumulate lines 80-82) only fires when the FIRST operand is the
 * broadcast one (`a` non-matching AND a->numel != 1). tensor_add(arg1,arg2)
 * binds e->arg1=arg1, so calling tensor_add(vec3, mat23) makes the rank-1
 * vector the broadcast operand and drives that arm.
 *
 * This block is F64-only (it uses tape_grad_load_d / tape_grad_add_d
 * throughout — no DT_F32 variant), so the inputs are plain F64 tensors built
 * with tensor_create (which COPIES, so stack arrays are fine) and grads are
 * read at TEST_TOL_TIGHT. Oracles are computed by hand from the inputs.
 *
 * Guarded BACKEND_TAPE: the file sits in a tape dir but is linked into every
 * backend's test binary; broadcast-backward grad semantics and storage differ
 * on torch/mlx, so the hand oracle is tape-specific.
 */

#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* a=[1,2,3] (rank-1, broadcast over axis 0), b=[[10,20,30],[40,50,60]] ([2,3],
   matches the result). r[i][j] = a[j] + b[i][j] = [[11,22,33],[41,52,63]].
   loss=sum (dr all ones). a broadcasts over the 2 rows, so da[j] = sum_i 1 = 2
   -> da = [2,2,2]; b matches r so db = all ones (fast loop, not the bcast arm).
   Drives the `do_a` general-broadcast backward arm (add.c:79-84, lines 80-82). */
Test(add_cov, bcast_backward_first_operand) {
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
	int sa[] = {3};
	int sb[] = {2, 3};
	TensorHandle a = tensor_create(ad, sa, 1, 1);
	TensorHandle b = tensor_create(bd, sb, 2, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle r = tensor_add(a, b);
	cr_assert_eq(tensor_numel(r), 6);
	double out[6];
	tensor_to_doubles(r, out);
	double expected_r[] = {11, 22, 33, 41, 52, 63};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expected_r[i], TEST_TOL_TIGHT, "r[%d] should be %.1f (got %.9f)",
		                   i, expected_r[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	/* da[j] summed over the broadcast row axis = 2 each. */
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(0, j), 2.0, TEST_TOL_TIGHT,
		                   "da[%d] should be 2 (got %.9f)", j, param_grad_item_at(0, j));
	/* db matches r: every element gets the unit seed. */
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), 1.0, TEST_TOL_TIGHT,
		                   "db[%d] should be 1 (got %.9f)", i, param_grad_item_at(1, i));
	param_clear();
}

/* Both operands broadcast into a [2,3] result, so BOTH the do_a and do_b
   stride-accumulate arms fire in one pass.
   a=[1,2,3] ([3], broadcast over rows); b=[[10],[20]] ([2,1], broadcast over
   cols). r[i][j] = a[j] + b[i] -> [[11,12,13],[21,22,23]].
   loss=sum: da[j] = sum_i 1 = 2 -> [2,2,2]; db[i] = sum_j 1 = 3 -> [3,3]. */
Test(add_cov, bcast_backward_both_operands) {
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {10.0, 20.0};
	int sa[] = {3};
	int sb[] = {2, 1};
	TensorHandle a = tensor_create(ad, sa, 1, 1);
	TensorHandle b = tensor_create(bd, sb, 2, 1);
	param_register("a", a);
	param_register("b", b);
	TensorHandle r = tensor_add(a, b);
	cr_assert_eq(tensor_numel(r), 6);
	double out[6];
	tensor_to_doubles(r, out);
	double expected_r[] = {11, 12, 13, 21, 22, 23};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expected_r[i], TEST_TOL_TIGHT, "r[%d] should be %.1f (got %.9f)",
		                   i, expected_r[i], out[i]);
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int j = 0; j < 3; j++)
		cr_assert_float_eq(param_grad_item_at(0, j), 2.0, TEST_TOL_TIGHT,
		                   "da[%d] should be 2 (got %.9f)", j, param_grad_item_at(0, j));
	for (int i = 0; i < 2; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), 3.0, TEST_TOL_TIGHT,
		                   "db[%d] should be 3 (got %.9f)", i, param_grad_item_at(1, i));
	param_clear();
}

#endif /* BACKEND_TAPE */

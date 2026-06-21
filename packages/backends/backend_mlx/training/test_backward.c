/* mlx-only Criterion suite for training/backward.cpp — the replay-based vjp.
 *
 * The common (tape-shared) op tests already lift mlx line coverage to
 * ~84%, but backward.cpp sat at 46% because nothing drove the
 * param-pool / mx::vjp / grad-distribution path under the mlx replay
 * model with a *variety* of ops, nor the env-gated paths
 * (DEBUG_NAN_TRAP, MLX_COMPILE).
 *
 * Each test registers params, builds a forward graph through the public
 * FFI, calls tensor_backward, and asserts the resulting grads
 * (param_grad_item_at). The graph variety (add / mul / matmul /
 * softmax / div / a non-param constant) exercises different replay arms
 * of the forward_fn closure (op_dispatch) plus the constant-pool
 * collection (a non-param tensor on the live tape becomes a vjp
 * constant input).
 *
 * Params/inputs use the F64 dtag (15) so the default mlx-cpu F64 path is
 * exercised; value asserts use TEST_TOL_TIGHT (1e-5 on mlx, since mlx
 * readback carries ~1e-6 error even on the F64 path).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* Heap-copy a stack array — streamed creators take ownership and free it. */

/* ---- early return: backward on a tensor with no tape entry (tape_idx<0) ---- */

Test(mlx_training_backward, leaf_const_no_tape_is_noop) {
	/* A freshly-created scalar that never feeds any op has tape_idx < 0.
	   tensor_backward must hit the early-return arm (backward.cpp 47-50)
	   and NOT crash / NOT touch params. */
	param_clear();
	TensorHandle a = tensor_create_scalar(3.0, 1);
	param_register("a", a);
	tensor_backward(a); /* loss->tape_idx < 0 -> early return */
	/* No grad was computed; nothing to assert beyond "did not crash". */
	cr_assert(1, "backward on a tapeless leaf returned cleanly");
	param_clear();
}

/* ---- core vjp: add ---- */

Test(mlx_training_backward, add_grads_both_one) {
	/* loss = sum(a + b); d/da = d/db = 1. Drives the param-pool build,
	   the forward_fn replay (OP_ADD + OP_SUM), mx::vjp, and the
	   grad-distribution loop. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {10.0, 20.0, 30.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 15);
	TensorHandle b = tensor_create_param_1d_streamed(3, hcopy(bd, 3), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_add(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT,
		                   "d(sum(a+b))/da[%d] should be 1.0 (got %.6f)", i,
		                   param_grad_item_at(0, i));
		cr_assert_float_eq(param_grad_item_at(1, i), 1.0, TEST_TOL_TIGHT,
		                   "d(sum(a+b))/db[%d] should be 1.0 (got %.6f)", i,
		                   param_grad_item_at(1, i));
	}
	param_clear();
}

/* ---- core vjp: mul (different replay arm) ---- */

Test(mlx_training_backward, mul_grads_are_other_operand) {
	/* loss = sum(a * b); d/da = b, d/db = a. */
	param_clear();
	double ad[] = {2.0, 3.0, 4.0};
	double bd[] = {5.0, 6.0, 7.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 15);
	TensorHandle b = tensor_create_param_1d_streamed(3, hcopy(bd, 3), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_mul(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), bd[i], TEST_TOL_TIGHT,
		                   "d(sum(a*b))/da[%d] should be b[%d]=%.1f (got %.6f)", i, i, bd[i],
		                   param_grad_item_at(0, i));
		cr_assert_float_eq(param_grad_item_at(1, i), ad[i], TEST_TOL_TIGHT,
		                   "d(sum(a*b))/db[%d] should be a[%d]=%.1f (got %.6f)", i, i, ad[i],
		                   param_grad_item_at(1, i));
	}
	param_clear();
}

/* ---- core vjp: matmul (OP_MM replay arm + 2-D shapes) ---- */

Test(mlx_training_backward, matmul_grad_is_row_sums) {
	/* loss = sum(a @ b), a:[2,2], b:[2,2].
	   d(sum(a@b))/da[i,k] = sum_j b[k,j]; d/db[k,j] = sum_i a[i,k]. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0}; /* [2,2] */
	double bd[] = {5.0, 6.0, 7.0, 8.0}; /* [2,2] */
	TensorHandle a = tensor_create_param_2d_streamed(2, 2, hcopy(ad, 4), 0, 15);
	TensorHandle b = tensor_create_param_2d_streamed(2, 2, hcopy(bd, 4), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_mm(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	/* d/da[i,k] = sum_j b[k,j]: row0 b -> 5+6=11, row1 b -> 7+8=15.
	   da = [[11,15],[11,15]] flattened. */
	double da_exp[] = {11.0, 15.0, 11.0, 15.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), da_exp[i], TEST_TOL_TIGHT,
		                   "d(sum(a@b))/da[%d] should be %.1f (got %.6f)", i, da_exp[i],
		                   param_grad_item_at(0, i));
	/* d/db[k,j] = sum_i a[i,k]: col0 a -> 1+3=4, col1 a -> 2+4=6.
	   db = [[4,4],[6,6]] flattened. */
	double db_exp[] = {4.0, 4.0, 6.0, 6.0};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(1, i), db_exp[i], TEST_TOL_TIGHT,
		                   "d(sum(a@b))/db[%d] should be %.1f (got %.6f)", i, db_exp[i],
		                   param_grad_item_at(1, i));
	param_clear();
}

/* ---- core vjp: softmax (OP_SOFTMAX_2D replay arm) ---- */

Test(mlx_training_backward, softmax_sum_grad_is_zero) {
	/* loss = sum(softmax(a)) == number of rows (each row sums to 1),
	   constant w.r.t. a, so d/da is ~0 everywhere. Exercises the
	   softmax replay + its vjp. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0, 4.0}; /* [2,2] */
	TensorHandle a = tensor_create_param_2d_streamed(2, 2, hcopy(ad, 4), 0, 15);
	param_register("a", a);
	TensorHandle s = tensor_softmax_2d(a);
	TensorHandle loss = tensor_sum(s);
	tensor_backward(loss);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, TEST_TOL_TIGHT,
		                   "d(sum(softmax(a)))/da[%d] should be ~0 (got %.6f)", i,
		                   param_grad_item_at(0, i));
	param_clear();
}

/* ---- core vjp: div (OP_DIV replay arm) ---- */

Test(mlx_training_backward, div_grad_numerator) {
	/* loss = sum(a / b); d/da = 1/b. b is registered too so both arms run. */
	param_clear();
	double ad[] = {6.0, 8.0};
	double bd[] = {2.0, 4.0};
	TensorHandle a = tensor_create_param_1d_streamed(2, hcopy(ad, 2), 0, 15);
	TensorHandle b = tensor_create_param_1d_streamed(2, hcopy(bd, 2), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_div(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	/* d/da[i] = 1/b[i] */
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.5, TEST_TOL_TIGHT, "d(sum(a/b))/da[0]=1/2");
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.25, TEST_TOL_TIGHT, "d(sum(a/b))/da[1]=1/4");
	/* d/db[i] = -a[i]/b[i]^2 = -6/4=-1.5, -8/16=-0.5 */
	cr_assert_float_eq(param_grad_item_at(1, 0), -1.5, TEST_TOL_TIGHT, "d(sum(a/b))/db[0]");
	cr_assert_float_eq(param_grad_item_at(1, 1), -0.5, TEST_TOL_TIGHT, "d(sum(a/b))/db[1]");
	param_clear();
}

/* ---- constant-pool path: a non-param tensor feeding the loss ---- */

Test(mlx_training_backward, nonparam_constant_in_pool) {
	/* k is created with requires_grad=0 and NOT registered, so it is a
	   live-tape tensor that is neither a param nor an index arg ->
	   add_const() collects it into `constants`, and it rides the
	   [params..., constants...] vjp-inputs vector (backward.cpp ~70-90,
	   127-135). Only the param grad is written back. */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double kd[] = {2.0, 2.0, 2.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 15);
	TensorHandle k =
	    tensor_create_1d_streamed(3, hcopy(kd, 3), 0, 0, 15); /* no grad, not a param */
	param_register("a", a);
	TensorHandle c = tensor_mul(a, k); /* loss = sum(a * k); d/da = k = 2 */
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 2.0, TEST_TOL_TIGHT,
		                   "d(sum(a*k))/da[%d] should be k=2 (got %.6f)", i,
		                   param_grad_item_at(0, i));
	param_clear();
}

/* ---- DEBUG_NAN_TRAP enabled, valid grads: walks param grads, no NaN ---- */

Test(mlx_training_backward, nan_trap_clean_grads) {
	/* With DEBUG_NAN_TRAP=1 and finite grads, the trap loop
	   (backward.cpp 164-194) walks every param grad, finds nan_count==0
	   / inf_count==0 for each, leaves any_nan==0, and skips the deep
	   forward-tape scan. Grads must still be correct. */
	setenv("DEBUG_NAN_TRAP", "1", 1);
	param_clear();
	double ad[] = {3.0, 5.0};
	double bd[] = {7.0, 9.0};
	TensorHandle a = tensor_create_param_1d_streamed(2, hcopy(ad, 2), 0, 15);
	TensorHandle b = tensor_create_param_1d_streamed(2, hcopy(bd, 2), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_add(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	for (int i = 0; i < 2; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TEST_TOL_TIGHT, "clean grad da[%d]=1", i);
		cr_assert_float_eq(param_grad_item_at(1, i), 1.0, TEST_TOL_TIGHT, "clean grad db[%d]=1", i);
	}
	param_clear();
	unsetenv("DEBUG_NAN_TRAP");
}

/* ---- MLX_COMPILE=1: compile-enabled vjp branch ---- */

Test(mlx_training_backward, compile_enabled_branch) {
	/* With MLX_COMPILE=1, tensor_backward takes the mx::compile path
	   (backward.cpp 138-141): increments g_compile_invocations and
	   compiles forward_vec before the vjp. The grads must match the
	   eager path. tensor_mlx_compile_invocations() must advance. */
	setenv("MLX_COMPILE", "1", 1);
	int before = tensor_mlx_compile_invocations();
	param_clear();
	double ad[] = {2.0, 3.0};
	double bd[] = {4.0, 5.0};
	TensorHandle a = tensor_create_param_1d_streamed(2, hcopy(ad, 2), 0, 15);
	TensorHandle b = tensor_create_param_1d_streamed(2, hcopy(bd, 2), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle c = tensor_mul(a, b);
	TensorHandle loss = tensor_sum(c);
	tensor_backward(loss);
	int after = tensor_mlx_compile_invocations();
	cr_assert_gt(after, before,
	             "MLX_COMPILE=1 should advance compile-invocation counter (%d -> %d)", before,
	             after);
	/* d(sum(a*b))/da = b */
	cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, TEST_TOL_TIGHT, "compile-path da[0]=b[0]");
	cr_assert_float_eq(param_grad_item_at(0, 1), 5.0, TEST_TOL_TIGHT, "compile-path da[1]=b[1]");
	param_clear();
	unsetenv("MLX_COMPILE");
}

/* ---- composite graph: chain several ops to exercise more replay arms ---- */

Test(mlx_training_backward, composite_chain) {
	/* loss = sum( (a + b) * a ) = sum(a^2 + a*b).
	   d/da = 2a + b ; d/db = a. Exercises add+mul+sum replay together
	   with a reused operand (a appears twice -> two tape consumers). */
	param_clear();
	double ad[] = {1.0, 2.0, 3.0};
	double bd[] = {4.0, 5.0, 6.0};
	TensorHandle a = tensor_create_param_1d_streamed(3, hcopy(ad, 3), 0, 15);
	TensorHandle b = tensor_create_param_1d_streamed(3, hcopy(bd, 3), 0, 15);
	param_register("a", a);
	param_register("b", b);
	TensorHandle s = tensor_add(a, b);
	TensorHandle p = tensor_mul(s, a);
	TensorHandle loss = tensor_sum(p);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		double da_exp = 2.0 * ad[i] + bd[i];
		cr_assert_float_eq(param_grad_item_at(0, i), da_exp, TEST_TOL_TIGHT,
		                   "d/da[%d] should be 2a+b=%.1f (got %.6f)", i, da_exp,
		                   param_grad_item_at(0, i));
		cr_assert_float_eq(param_grad_item_at(1, i), ad[i], TEST_TOL_TIGHT,
		                   "d/db[%d] should be a=%.1f (got %.6f)", i, ad[i],
		                   param_grad_item_at(1, i));
	}
	param_clear();
}

#endif /* BACKEND_MLX */

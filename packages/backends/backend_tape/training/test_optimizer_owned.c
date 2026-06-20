/* Criterion suite for owned-set optimizer scoping (Option B true-skip).
 *
 * optimizer_own_param(opt, name) adds one EXACT name to the optimizer's
 * owned-set; once non-empty, the optimizer's step + grad-clip touch ONLY
 * owned params. An empty owned-set (default) manages every registered param.
 *
 * RED before Option B: ownership was a prefix string that the typed surface
 * never set, so opt_owns_param returned 1 for every param — `own_b` below was
 * stepped to 0.8 too (LR-0-on-complement was the only thing scoping restrictTo,
 * and that doesn't skip the C step loop). True-skip leaves own_b at 1.0.
 *
 * Colocated under backend_tape/ but compiled into all three backend test
 * binaries via the test_*.c glob; it calls the public renamed
 * `optimizer_own_param`, so the same oracle runs on tape/torch/mlx.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

/* loss = a*a + b*b, so grad(a)=2a, grad(b)=2b; SGD lr=0.1 => owned param
   moves 1 -> 1 - 0.1*2 = 0.8. */
Test(training_optimizer_owned, owned_step_skips_unowned) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	TensorHandle b = tensor_create_scalar(1.0, 1);
	param_register("own_a", a);
	param_register("own_b", b);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	optimizer_own_param(opt, "own_a"); /* owns ONLY own_a */
	TensorHandle loss = tensor_add(tensor_mul(a, a), tensor_mul(b, b));
	native_train_step(opt, 0, 0.0, loss, 2.0);
	cr_assert_float_eq(tensor_item(a), 0.8, TEST_TOL_RELAXED,
	                   "own_a is owned -> stepped to 0.8 (got %.9f)", tensor_item(a));
	cr_assert_float_eq(tensor_item(b), 1.0, TEST_TOL_RELAXED,
	                   "own_b NOT owned -> untouched at 1.0 (got %.9f)", tensor_item(b));
	optimizer_free(opt);
	param_clear();
}

/* Empty owned-set (default) manages every param — the single-optimizer path. */
Test(training_optimizer_owned, empty_owned_manages_all) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	TensorHandle b = tensor_create_scalar(1.0, 1);
	param_register("all_a", a);
	param_register("all_b", b);
	OptimizerHandle opt = optimizer_create_sgd(0.1); /* no own_param => owns all */
	TensorHandle loss = tensor_add(tensor_mul(a, a), tensor_mul(b, b));
	native_train_step(opt, 0, 0.0, loss, 2.0);
	cr_assert_float_eq(tensor_item(a), 0.8, TEST_TOL_RELAXED, "all_a stepped (got %.9f)",
	                   tensor_item(a));
	cr_assert_float_eq(tensor_item(b), 0.8, TEST_TOL_RELAXED, "all_b stepped (got %.9f)",
	                   tensor_item(b));
	optimizer_free(opt);
	param_clear();
}

/* mlx-only Criterion suite for training/autograd.cpp — the non-replay
 * autograd surface (grad accessor, zero/detach/with_grad/requires_grad
 * predicates, plus the no_grad + epoch generation-scoped sweep).
 *
 * autograd.cpp has no prior coverage; this file pins its public entry
 * points and the edge arms inside mlx_sweep_generation (the static
 * sweep shared by tensor_no_grad_end + tensor_epoch_end). Each Test
 * targets a specific arm:
 *
 *   - tensor_requires_grad        true + false predicate
 *   - tensor_set_requires_grad    0->1 and 1->0 toggle
 *   - tensor_detach               clone w/ requires_grad=false, data kept
 *   - tensor_with_grad            clone w/ requires_grad=true, data kept
 *   - tensor_grad (no-grad arm)   has_grad false -> nullptr short-circuit
 *   - tensor_grad (contig arm)    has_grad true  -> contiguous grad copy
 *   - tensor_zero_grad (true arm) has_grad true  -> grad rebuilt as zeros
 *   - tensor_zero_grad (skip arm) has_grad false -> no-op
 *   - no_grad nesting             begin(d==0)/begin(d>0)/end(d>0 return)/
 *                                 end(d==0 sweep) + end-at-d==0 guard-false
 *   - epoch scope                 epoch_end(empty-stack return),
 *                                 epoch_begin/epoch_end sweep:
 *                                   survivor arm (registered param, rc>0)
 *                                   has_grad eval arm (param after backward)
 *                                   husk arm  (rc==1 & create_id>=block)
 *                                   delete arm (rc==0 orphan)
 *
 * Refcount note: in a pure-C test there is no Idris guardian wrap, so a
 * freshly created Tensor has refcount 0. tape_append, however, retains
 * its result (tape.cpp), so any tensor produced by a forward op during a
 * backward is rc>=1 — that's why the sweep tests below never strand a
 * tape-referenced pointer. The rc==0 delete arm is exercised only with an
 * orphan scalar that nothing else references after the sweep.
 *
 * dtag 15 == F64; value asserts use TEST_TOL_TIGHT (1e-5 on mlx) except
 * exact-zero checks (0.0).
 */

#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* ---- tensor_requires_grad: true + false predicate ---- */

Test(autograd_cov, requires_grad_predicate) {
	TensorHandle a = tensor_create_scalar(2.0, 1);
	TensorHandle b = tensor_create_scalar(2.0, 0);
	cr_assert_eq(tensor_requires_grad(a), 1, "requires_grad of rg=1 scalar should be 1");
	cr_assert_eq(tensor_requires_grad(b), 0, "requires_grad of rg=0 scalar should be 0");
}

/* ---- tensor_set_requires_grad: toggle both ways ---- */

Test(autograd_cov, set_requires_grad_toggles) {
	TensorHandle a = tensor_create_scalar(2.0, 0);
	cr_assert_eq(tensor_requires_grad(a), 0, "starts at 0");
	tensor_set_requires_grad(a, 1);
	cr_assert_eq(tensor_requires_grad(a), 1, "set to 1");
	tensor_set_requires_grad(a, 0);
	cr_assert_eq(tensor_requires_grad(a), 0, "set back to 0");
}

/* ---- tensor_detach: clone, requires_grad=false, data preserved ---- */

Test(autograd_cov, detach_clears_requires_grad) {
	TensorHandle a = tensor_create_scalar(3.5, 1);
	TensorHandle d = tensor_detach(a);
	cr_assert_eq(tensor_requires_grad(d), 0, "detached tensor must have requires_grad=false");
	double buf[1];
	tensor_to_doubles(d, buf);
	cr_assert_float_eq(buf[0], 3.5, TEST_TOL_TIGHT, "detach must preserve data (got %.6f)", buf[0]);
}

/* ---- tensor_with_grad: clone, requires_grad=true, data preserved ---- */

Test(autograd_cov, with_grad_sets_requires_grad) {
	TensorHandle a = tensor_create_scalar(2.5, 0);
	TensorHandle w = tensor_with_grad(a);
	cr_assert_eq(tensor_requires_grad(w), 1, "with_grad tensor must have requires_grad=true");
	double buf[1];
	tensor_to_doubles(w, buf);
	cr_assert_float_eq(buf[0], 2.5, TEST_TOL_TIGHT, "with_grad must preserve data (got %.6f)",
	                   buf[0]);
}

/* ---- tensor_grad: no-grad short-circuit arm ---- */

Test(autograd_cov, grad_null_when_no_backward) {
	/* has_grad is false before any backward -> tensor_grad returns the
	   nullptr branch rather than building a contiguous copy. */
	TensorHandle a = tensor_create_scalar(2.0, 1);
	TensorHandle g = tensor_grad(a);
	cr_assert_null(g, "tensor_grad before backward should be NULL (has_grad false)");
}

/* ---- tensor_grad: contiguous-copy arm after backward ---- */

Test(autograd_cov, grad_contiguous_after_backward) {
	/* loss = sum(p*p); d/dp[i] = 2*p[i]. has_grad is true after backward,
	   so tensor_grad takes the mx::contiguous + eval + new Tensor branch. */
	param_clear();
	double init[] = {3.0, -4.0, 5.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), 0, 15);
	param_register("p", p);
	TensorHandle loss = tensor_sum(tensor_mul(p, p));
	tensor_backward(loss);

	TensorHandle g = tensor_grad(p);
	cr_assert_not_null(g, "tensor_grad after backward should be non-NULL");
	double buf[3];
	tensor_to_doubles(g, buf);
	double expect[] = {6.0, -8.0, 10.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], expect[i], TEST_TOL_TIGHT,
		                   "grad[%d] = 2*p should be %.1f (got %.6f)", i, expect[i], buf[i]);
	param_clear();
}

/* ---- tensor_zero_grad: has_grad-true arm rebuilds grad as zeros ---- */

Test(autograd_cov, zero_grad_clears_after_backward) {
	param_clear();
	double init[] = {2.0, -3.0};
	TensorHandle p = tensor_create_param_1d_streamed(2, hcopy(init, 2), 0, 15);
	param_register("p", p);
	TensorHandle loss = tensor_sum(tensor_mul(p, p));
	tensor_backward(loss); /* grads = 2*p = {4, -6} */
	cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, TEST_TOL_TIGHT, "pre-zero grad[0]");

	tensor_zero_grad(p); /* has_grad true -> grad := zeros */
	TensorHandle g = tensor_grad(p);
	cr_assert_not_null(g, "grad still present after zero_grad (has_grad stays set)");
	double buf[2];
	tensor_to_doubles(g, buf);
	cr_assert_float_eq(buf[0], 0.0, TEST_TOL_TIGHT, "zeroed grad[0] (got %.6f)", buf[0]);
	cr_assert_float_eq(buf[1], 0.0, TEST_TOL_TIGHT, "zeroed grad[1] (got %.6f)", buf[1]);
	param_clear();
}

/* ---- tensor_zero_grad: has_grad-false skip arm (no-op, no crash) ---- */

Test(autograd_cov, zero_grad_noop_without_grad) {
	TensorHandle a = tensor_create_scalar(5.0, 1);
	tensor_zero_grad(a); /* has_grad false -> the if-body is skipped */
	cr_assert_eq(tensor_requires_grad(a), 1, "zero_grad must not disturb requires_grad");
}

/* ---- no_grad scope: nesting + outermost-only sweep + guard-false end ---- */

Test(autograd_cov, no_grad_scope_nesting) {
	/* Arms hit, in order:
	   - end at depth 0: first `if (depth>0)` guard false, then sweep
	   - begin at depth 0: capture g_nograd_block_start
	   - begin at depth 1: plain increment
	   - end at depth 2->1: second `if (depth>0)` return (no sweep)
	   - end at depth 1->0: sweep (outermost). */
	param_clear();
	tensor_no_grad_end(); /* depth already 0: guard-false then sweep over empty set */

	tensor_no_grad_begin(); /* 0 -> 1 */
	tensor_no_grad_begin(); /* 1 -> 2 */
	/* tensor created inside no_grad: tape_append is gated so this stays an
	   rc==0 orphan; the outermost end's sweep reclaims it. Not touched after. */
	TensorHandle inside = tensor_create_scalar(1.0, 0);
	(void)inside;
	tensor_no_grad_end(); /* 2 -> 1: returns before sweep */
	tensor_no_grad_end(); /* 1 -> 0: sweep */
	param_clear();
}

/* ---- epoch scope: empty-stack return + sweep survivor/has_grad arms ---- */

Test(autograd_cov, epoch_scope_sweep_survivor_and_grad) {
	param_clear();
	tensor_epoch_end(); /* g_gen_stack empty -> early return arm */

	double init[] = {3.0, -4.0};
	TensorHandle p = tensor_create_param_1d_streamed(2, hcopy(init, 2), 0, 15);
	param_register("p", p);

	tensor_epoch_begin(); /* push block_start = current create count */
	TensorHandle loss = tensor_sum(tensor_mul(p, p));
	tensor_backward(loss); /* p->has_grad true; grads = 2*p */
	/* sweep arms exercised: has_grad eval (p+grad pushed), survivor (p rc>0
	   and the rc>=2 mul result), husk (sum result is rc==1 & post-block). */
	tensor_epoch_end();

	/* p survives the sweep with its grad intact. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 6.0, TEST_TOL_TIGHT, "grad[0]=2*3 (got %.6f)",
	                   param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), -8.0, TEST_TOL_TIGHT, "grad[1]=2*-4 (got %.6f)",
	                   param_grad_item_at(0, 1));
	param_clear();
}

/* ---- epoch sweep: rc==0 delete arm ---- */

Test(autograd_cov, epoch_sweep_deletes_orphan) {
	/* An unregistered, unwrapped scalar created after epoch_begin is rc==0,
	   so the sweep takes the `delete t` arm. Not referenced after the end. */
	param_clear();
	tensor_epoch_begin();
	TensorHandle orphan = tensor_create_scalar(9.0, 0);
	(void)orphan;
	tensor_epoch_end();
	param_clear();
}

/* ---- epoch sweep: rc==1 husk arm ---- */

Test(autograd_cov, epoch_sweep_keeps_husk) {
	/* A scalar retained to rc==1 after epoch_begin matches the husk
	   predicate (refcount==1 && create_id>=block_start): the sweep replaces
	   its heavy buffer with the shared empty scalar and keeps the husk in
	   all_tensors rather than deleting it. We only assert it does not crash;
	   the husk's data is intentionally not read. */
	param_clear();
	tensor_epoch_begin();
	TensorHandle h = tensor_create_scalar(7.0, 0);
	tensor_retain_handle(h); /* rc 0 -> 1 */
	tensor_epoch_end();      /* husk arm */
	(void)h;
	param_clear();
}

#endif /* BACKEND_MLX */

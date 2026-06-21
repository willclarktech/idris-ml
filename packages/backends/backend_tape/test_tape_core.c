/* Criterion suite for backend_tape/tape.c — the TypedArena<TapeEntry>
 * machinery + tape_append / tape_reset + the no_grad mechanism, plus the
 * multi-chunk arena growth / reset-reuse path past TAPE_CHUNK_SIZE.
 *
 * Driven entirely through the public FFI (backend.h): each Criterion test
 * forks into a fresh child with an empty tape, so we can rely on a clean
 * arena per test.
 *
 *   - no_grad path in tape_append (skip-entry + scratch-dummy return) is
 *     exercised by tensor_no_grad_begin/end around a real op.
 *   - typed_arena_at / tape_at are walked by tensor_backward over a
 *     multi-entry tape.
 *   - tape_reset's per-op heap-free arms (OP_DROPOUT mask, OP_EMBEDDING
 *     indices) are reached by running those ops then backend_reset_for_eval
 *     (which calls tape_reset).
 *   - the TapeEntry arena's multi-chunk grow / reset / reuse arms are driven
 *     by a long chain of scalar ops crossing the 65,536-entry boundary.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* no_grad: an op run between tensor_no_grad_begin/end produces a correct
   forward value but is not grad-tracked. This drives the early-return arm
   of tape_append (result->requires_grad=0, tape_idx=-1, scratch dummy). */
Test(tape_core, no_grad_skips_tape_but_computes) {
	param_clear();
	double xd[] = {2.0, 3.0};
	int s[] = {2};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);

	tensor_no_grad_begin();
	TensorHandle y = tensor_mul(x, x); /* tape_append hits no_grad arm */
	TensorHandle loss = tensor_sum(y); /* another no_grad append */
	tensor_no_grad_end();

	/* Forward value is still correct: 2^2 + 3^2 = 13. */
	cr_assert_float_eq(tensor_item(loss), 13.0, 1e-12, "no_grad sum(x*x) = 13");

	/* Backward must NOT populate x's grad — the no_grad result has
	   tape_idx=-1, so the chain is severed. grad stays 0. */
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "no_grad: x[0] grad stays 0");
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, 1e-12, "no_grad: x[1] grad stays 0");
	param_clear();
}

/* Nested no_grad scopes: depth is a counter, so an inner end must not
   re-enable tracking while the outer scope is still open. */
Test(tape_core, no_grad_nests) {
	param_clear();
	double xd[] = {4.0};
	int s[] = {1};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);

	tensor_no_grad_begin();
	tensor_no_grad_begin();
	tensor_no_grad_end(); /* depth 2 -> 1: still inside no_grad */
	TensorHandle y = tensor_mul(x, x);
	TensorHandle loss = tensor_sum(y);
	tensor_no_grad_end(); /* depth 1 -> 0 */

	cr_assert_float_eq(tensor_item(loss), 16.0, 1e-12, "nested no_grad still computes 16");
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "still-in-no_grad: grad 0");
	param_clear();
}

/* A multi-op backward walks the tape via tape_at -> typed_arena_at for
   every entry. d(sum((x*x)*x))/dx = 3x^2. */
Test(tape_core, backward_walks_tape) {
	param_clear();
	double xd[] = {2.0, 3.0};
	int s[] = {2};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);

	TensorHandle x2 = tensor_mul(x, x);
	TensorHandle x3 = tensor_mul(x2, x);
	TensorHandle loss = tensor_sum(x3);

	cr_assert_float_eq(tensor_item(loss), 8.0 + 27.0, 1e-12, "sum(x^3) = 35");
	tensor_backward(loss);
	/* 3x^2: 3*4=12, 3*9=27 */
	cr_assert_float_eq(param_grad_item_at(0, 0), 12.0, 1e-12, "grad x[0] = 3*4");
	cr_assert_float_eq(param_grad_item_at(0, 1), 27.0, 1e-12, "grad x[1] = 3*9");
	param_clear();
}

/* tape_reset frees the OP_DROPOUT mask. Run dropout (training=1 allocates
   the mask), then backend_reset_for_eval -> tape_reset walks the entry and
   frees meta->mask. A clean run (no leak / double-free) covers the arm. */
Test(tape_core, reset_frees_dropout_mask) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	int s[] = {4};
	TensorHandle x = tensor_create(xd, s, 1, 1);
	param_register("x", x);
	TensorHandle d = tensor_dropout(x, 0.5, 1, 123u); /* OP_DROPOUT, mask heap-alloc */
	TensorHandle loss = tensor_sum(d);
	tensor_backward(loss);
	backend_reset_for_eval(); /* tape_reset -> free(meta->mask) */
	/* A second forward after reset must still work (mask wasn't double-freed). */
	TensorHandle d2 = tensor_dropout(x, 0.5, 1, 123u);
	cr_assert_eq(tensor_numel(d2), 4, "post-reset dropout still produces a [4] tensor");
	param_clear();
}

/* tape_reset frees the OP_EMBEDDING indices array. */
Test(tape_core, reset_frees_embedding_indices) {
	param_clear();
	/* weight: 3 rows x 2 cols */
	double wd[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	int ws[] = {3, 2};
	TensorHandle w = tensor_create(wd, ws, 2, 1);
	param_register("w", w);
	/* indices [2]: gather rows 0 and 2 */
	double idxd[] = {0.0, 2.0};
	int is[] = {2};
	TensorHandle idx = tensor_create(idxd, is, 1, 0);
	TensorHandle emb = tensor_embedding(w, idx, 2, 2); /* OP_EMBEDDING, indices heap-alloc */
	TensorHandle loss = tensor_sum(emb);
	cr_assert_float_eq(tensor_item(loss), 0.0 + 1.0 + 4.0 + 5.0, 1e-12, "embedding sum = 10");
	tensor_backward(loss);
	backend_reset_for_eval(); /* tape_reset -> free(meta->indices) */
	param_clear();
}

/* tape_reset frees the OP_LAYER_NORM_2D x_hat + rstd arrays. */
Test(tape_core, reset_frees_layer_norm_meta) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0}; /* [2,2] */
	int xs[] = {2, 2};
	TensorHandle x = tensor_create(xd, xs, 2, 1);
	param_register("x", x);
	double gd[] = {1.0, 1.0};
	double bd[] = {0.0, 0.0};
	int gs[] = {2};
	TensorHandle gamma = tensor_create(gd, gs, 1, 0);
	TensorHandle bias = tensor_create(bd, gs, 1, 0);
	TensorHandle ln = tensor_layer_norm_2d(x, gamma, bias, 1e-5); /* OP_LAYER_NORM_2D */
	TensorHandle loss = tensor_sum(ln);
	tensor_backward(loss);
	backend_reset_for_eval(); /* tape_reset -> free(meta->x_hat/rstd) */
	param_clear();
}

/* tape_reset frees the OP_RMS_NORM_2D x_hat + rstd arrays. */
Test(tape_core, reset_frees_rms_norm_meta) {
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0}; /* [2,2] */
	int xs[] = {2, 2};
	TensorHandle x = tensor_create(xd, xs, 2, 1);
	param_register("x", x);
	double wd[] = {1.0, 1.0};
	int ws[] = {2};
	TensorHandle w = tensor_create(wd, ws, 1, 0);
	TensorHandle rn = tensor_rms_norm_2d(x, w, 1e-6); /* OP_RMS_NORM_2D */
	TensorHandle loss = tensor_sum(rn);
	tensor_backward(loss);
	backend_reset_for_eval(); /* tape_reset -> free(meta->x_hat/rstd) */
	param_clear();
}

/* tape_reset frees the OP_BATCH_NORM x_hat + rstd arrays. */
Test(tape_core, reset_frees_batch_norm_meta) {
	param_clear();
	/* input [C=2, spatial=2] = 4 elems */
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	int xs[] = {2, 2};
	TensorHandle x = tensor_create(xd, xs, 2, 1);
	param_register("x", x);
	double gd[] = {1.0, 1.0}, bd[] = {0.0, 0.0};
	double rm[] = {0.0, 0.0}, rv[] = {1.0, 1.0};
	int cs[] = {2};
	TensorHandle gamma = tensor_create(gd, cs, 1, 0);
	TensorHandle beta = tensor_create(bd, cs, 1, 0);
	TensorHandle rmean = tensor_create(rm, cs, 1, 0);
	TensorHandle rvar = tensor_create(rv, cs, 1, 0);
	TensorHandle bn =
	    tensor_batch_norm(x, gamma, beta, rmean, rvar, 2, 2, 1, 0.1, 1e-5); /* OP_BATCH_NORM */
	TensorHandle loss = tensor_sum(bn);
	tensor_backward(loss);
	backend_reset_for_eval(); /* tape_reset -> free(meta->x_hat/rstd) */
	param_clear();
}

/* tape_reset frees the OP_MAX_POOL1D max_indices array. */
Test(tape_core, reset_frees_max_pool1d_indices) {
	param_clear();
	/* input [C=1, L=4] */
	double xd[] = {1.0, 3.0, 2.0, 4.0};
	int xs[] = {1, 4};
	TensorHandle x = tensor_create(xd, xs, 2, 1);
	param_register("x", x);
	TensorHandle mp =
	    tensor_max_pool1d(x, 2, 2); /* OP_MAX_POOL1D -> [1,2]: max(1,3)=3, max(2,4)=4 */
	TensorHandle loss = tensor_sum(mp);
	cr_assert_float_eq(tensor_item(loss), 3.0 + 4.0, 1e-12, "max_pool1d sum = 7");
	tensor_backward(loss);
	backend_reset_for_eval(); /* tape_reset -> free(meta->max_indices) */
	param_clear();
}

/* ---- TapeEntry arena multi-chunk growth coverage (from
   test_arena_growth_cov_tape.c) ---------------------------------------- */

extern void tape_reset(void);

/* TAPE_CHUNK_SIZE = 1 << 16 = 65536; go comfortably past it. */
#define N_OPS 65600

Test(arena_growth_cov, multi_chunk_grow_reset_reuse) {
	/* Pass 1: > 65536 tracked ops force a second chunk allocation. */
	param_clear();
	TensorHandle x = tensor_create_scalar(0.0, /*requires_grad=*/1);
	param_register("x", x);
	TensorHandle y = x;
	for (int i = 0; i < N_OPS; i++)
		y = tensor_add_scalar(y, 1.0); /* each appends one OP_ADD_SCALAR entry */
	cr_assert_float_eq(tensor_item(y), (double)N_OPS, TEST_TOL_TIGHT,
	                   "sum of %d unit adds (got %.1f)", N_OPS, tensor_item(y));
	tensor_backward(y); /* d(x + N)/dx = 1, across the chunk boundary */
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_TIGHT,
	                   "grad through %d-op chain should be 1 (got %.9f)", N_OPS,
	                   param_grad_item_at(0, 0));

	/* Reset a >65536-entry tape: exercises the multi-chunk teardown walk. */
	tape_reset();

	/* Pass 2: re-grow past the boundary — the second chunk already exists,
	   so the append takes the reuse-existing-next-chunk arm. */
	param_clear();
	TensorHandle x2 = tensor_create_scalar(0.0, /*requires_grad=*/1);
	param_register("x2", x2);
	TensorHandle y2 = x2;
	for (int i = 0; i < N_OPS; i++)
		y2 = tensor_add_scalar(y2, 2.0);
	cr_assert_float_eq(tensor_item(y2), (double)N_OPS * 2.0, TEST_TOL_TIGHT,
	                   "sum of %d twos (got %.1f)", N_OPS, tensor_item(y2));
	tensor_backward(y2);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, TEST_TOL_TIGHT,
	                   "grad after chunk reuse should be 1 (got %.9f)", param_grad_item_at(0, 0));
	param_clear();
}

#endif /* BACKEND_TAPE */

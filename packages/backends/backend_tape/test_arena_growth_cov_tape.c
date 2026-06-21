/* tape arena multi-chunk growth coverage (tape.c).
 *
 * The TapeEntry TypedArena allocates in 65,536-entry chunks. The chunk
 * machinery — new-chunk allocation (typed_arena_append, the malloc/calloc
 * arm), the reuse-existing-next-chunk arm after a reset, and the
 * multi-chunk walk in tape_reset — only fires past TAPE_CHUNK_SIZE entries,
 * which no other test reaches. A long chain of tape-tracked scalar ops
 * crosses the boundary cheaply (C-only, no Idris elaboration).
 */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

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

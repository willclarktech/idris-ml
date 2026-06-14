/* Regression guard for the tape arena allocator's recycle-undersized-chunk
 * heap-buffer-overflow (campaign 2026-06-17 MNIST "invalid memory reference").
 *
 * `arena_alloc` (arena.c) grows by appending chunks; after `arena_reset`
 * rewinds the bump pointer to the head, a later allocation that overflows the
 * current chunk *recycles the pre-allocated `next` chunk* — but historically
 * did so WITHOUT checking that the recycled chunk was large enough. A request
 * bigger than the recycled chunk's capacity then overran it on the `memcpy`
 * in `make_tensor`. MNIST's conv1 output ([64,16,24,24] = 4.5 MB) is the only
 * intermediate in any example that exceeds the 4 MB default chunk size
 * (ARENA_INIT_SIZE), so it was the only example that tripped it — manifesting
 * downstream as a garbage `e->result` (overwritten Tensor header) in
 * tensor_backward.
 *
 * This drives the exact chain through the public tensor API (each Criterion
 * test runs in a forked child → a fresh, empty arena):
 *   1. small + ~3.84 MB tensors → fill the 4 MB head chunk
 *   2. a ~0.48 MB tensor → spills, allocating a second 4 MB `next` chunk
 *   3. tape_reset() → arena_reset rewinds to head; both chunks persist
 *   4. a >4 MB tensor → overflows head, recycles the 4 MB `next` chunk
 *      → RED: ASan heap-buffer-overflow in make_tensor (memcpy)
 *
 * Tape-only: the bump arena is a tape-backend internal.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"

#ifdef BACKEND_TAPE

/* Sizes in doubles, against ARENA_INIT_SIZE = (1<<22) bytes = 524288 doubles. */
Test(arena_recycle, oversized_request_does_not_overflow_recycled_chunk) {
	param_clear();

	int n_small = 1, n_big = 480000, n_spill = 60000, n_over = 525000;
	double* small = (double*)calloc((size_t)n_small, sizeof(double));
	double* big = (double*)calloc((size_t)n_big, sizeof(double));
	double* spill = (double*)calloc((size_t)n_spill, sizeof(double));
	double* over = (double*)malloc((size_t)n_over * sizeof(double));
	for (int i = 0; i < n_over; i++)
		over[i] = (double)(i % 7) - 3.0;

	int s_small[1] = {n_small}, s_big[1] = {n_big}, s_spill[1] = {n_spill}, s_over[1] = {n_over};

	/* 1-2: build a head(4MB) -> next(4MB) chain. */
	tensor_create(small, s_small, 1, 0);
	tensor_create(big, s_big, 1, 0);     /* head now ~3.84 MB used (< 4 MB) */
	tensor_create(spill, s_spill, 1, 0); /* spills -> second 4 MB chunk */

	/* 3: rewind the bump pointer; both chunks stay in the chain.
	   backend_reset_for_eval → tape_reset → arena_reset (no params registered,
	   so the re-registration loop is a no-op). */
	backend_reset_for_eval();

	/* 4: >4 MB request. Overflows head, must NOT memcpy into the 4 MB
	   recycled `next` chunk. Pre-fix: ASan heap-buffer-overflow here. */
	TensorHandle z = tensor_create(over, s_over, 1, 0);

	/* Read first + last element back: confirms the 4.2 MB landed in an
	   adequately-sized buffer (no clobber, no OOB). */
	cr_assert_float_eq(tensor_item_1d(z, 0), over[0], 1e-12, "z[0]");
	cr_assert_float_eq(tensor_item_1d(z, n_over - 1), over[n_over - 1], 1e-12, "z[last]");

	free(small);
	free(big);
	free(spill);
	free(over);
	param_clear();
}

#endif /* BACKEND_TAPE */

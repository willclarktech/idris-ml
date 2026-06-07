/* linear/shape/narrow.c — slice along an axis.
 *
 * Forward:
 *   - rank=1 dim=0: shared-storage view via byte-offset pointer (cheap).
 *   - rank=2 dim=0: shared-storage view (contiguous rows).
 *   - rank=2 dim=1: COPY into a fresh arena buffer (slice columns are
 *     non-contiguous in the source; no stride field on Tensor, so a
 *     view-with-stride would require threading LDA through every
 *     downstream BLAS call).
 * Backward: scatter grad back to parent, dispatched on rank + which
 * dim was sliced (inferred from shape comparison; tape entry only
 * carries one scalar `start`).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
	Tensor* t = (Tensor*)h;
	size_t esz = tape_elem_size(t->dtype_tag);
	Tensor* r = arena_alloc(sizeof(Tensor));
	memset(r, 0, sizeof(Tensor));
	r->requires_grad = t->requires_grad;
	r->tape_idx = -1;
	r->dtype_tag = t->dtype_tag;

	if (t->rank == 1 && dim == 0) {
		/* 1D view. */
		r->data = (char*)t->data + (size_t)start * esz;
		r->shape = arena_alloc(sizeof(int));
		r->shape[0] = len;
		r->rank = 1;
		r->numel = len;
	} else if (t->rank == 2 && dim == 0) {
		/* 2D axis-0 narrow: contiguous rows, so a view works. */
		int cols = t->shape[1];
		r->data = (char*)t->data + (size_t)start * (size_t)cols * esz;
		r->shape = arena_alloc(2 * sizeof(int));
		r->shape[0] = len;
		r->shape[1] = cols;
		r->rank = 2;
		r->numel = len * cols;
	} else if (t->rank == 2 && dim == 1) {
		/* 2D axis-1 narrow: source slices are non-contiguous, so copy
		 * row-by-row into a fresh arena buffer. Downstream BLAS calls
		 * see a normal contiguous matrix. */
		int rows = t->shape[0];
		int cols = t->shape[1];
		char* dst = arena_alloc((size_t)rows * (size_t)len * esz);
		char* src = (char*)t->data;
		for (int row = 0; row < rows; row++) {
			memcpy(dst + (size_t)row * (size_t)len * esz,
			       src + ((size_t)row * (size_t)cols + (size_t)start) * esz, (size_t)len * esz);
		}
		r->data = dst;
		r->shape = arena_alloc(2 * sizeof(int));
		r->shape[0] = rows;
		r->shape[1] = len;
		r->rank = 2;
		r->numel = rows * len;
	} else if (t->rank == 3 && dim == 0) {
		/* 3D axis-0 narrow: contiguous pages, view works. */
		int b = t->shape[1];
		int c = t->shape[2];
		r->data = (char*)t->data + (size_t)start * (size_t)b * (size_t)c * esz;
		r->shape = arena_alloc(3 * sizeof(int));
		r->shape[0] = len;
		r->shape[1] = b;
		r->shape[2] = c;
		r->rank = 3;
		r->numel = len * b * c;
	} else if (t->rank == 3 && dim == 2) {
		/* 3D axis-2 narrow: innermost slice non-contiguous across the middle
		 * axis. Copy a*b slabs of `len * esz` bytes each. Used by
		 * `applyRopeAllHeads` to split the per-head dim into halves on
		 * a [seq, numHeads, headDim] view in one pass. */
		int a = t->shape[0];
		int b = t->shape[1];
		int c = t->shape[2];
		size_t slab = (size_t)len * esz;
		char* dst = arena_alloc((size_t)a * (size_t)b * slab);
		char* src = (char*)t->data;
		for (int i = 0; i < a; i++) {
			for (int j = 0; j < b; j++) {
				size_t out_off = ((size_t)i * (size_t)b + (size_t)j) * slab;
				size_t in_off =
				    (((size_t)i * (size_t)b + (size_t)j) * (size_t)c + (size_t)start) * esz;
				memcpy(dst + out_off, src + in_off, slab);
			}
		}
		r->data = dst;
		r->shape = arena_alloc(3 * sizeof(int));
		r->shape[0] = a;
		r->shape[1] = b;
		r->shape[2] = len;
		r->rank = 3;
		r->numel = a * b * len;
	} else {
		fprintf(stderr,
		        "tape tensor_narrow: unsupported (rank=%d, dim=%d). "
		        "Supported: rank=1+dim=0, rank=2+dim=0/1, rank=3+dim=0/2.\n",
		        t->rank, dim);
		// NOLINTNEXTLINE(misc-include-cleaner): macOS SDK: abort via _abort.h umbrella
		abort();
	}

	if (r->requires_grad) tape_append(OP_NARROW, r, t, NULL, (double)start);
	return r;
}

static void tape_backward_narrow(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	int start = (int)e->scalar_arg;
	ensure_grad(r);
	if (!a) return;
	ensure_grad(a);

	/* Infer which dim was narrowed from a/r shape comparison. Grad
	 * accumulation routes through tape_grad_*_d so F32 buffers narrow
	 * on store / widen on load. */
	if (a->rank == 1) {
		/* 1D narrow: contiguous segment starting at `start`. */
		for (int j = 0; j < r->numel; j++)
			tape_grad_add_d(a, start + j, tape_grad_load_d(r, j));
	} else if (a->rank == 2 && r->rank == 2 && r->shape[1] == a->shape[1]) {
		/* 2D axis-0 narrow: contiguous block of `len` rows starting
		 * at row `start`. */
		int cols = a->shape[1];
		for (int row = 0; row < r->shape[0]; row++)
			for (int col = 0; col < cols; col++)
				tape_grad_add_d(a, (start + row) * cols + col,
				                tape_grad_load_d(r, row * cols + col));
	} else if (a->rank == 2 && r->rank == 2 && r->shape[0] == a->shape[0]) {
		/* 2D axis-1 narrow: scatter columns back to parent. */
		int parent_cols = a->shape[1];
		int slice_cols = r->shape[1];
		for (int row = 0; row < r->shape[0]; row++)
			for (int col = 0; col < slice_cols; col++)
				tape_grad_add_d(a, row * parent_cols + start + col,
				                tape_grad_load_d(r, row * slice_cols + col));
	} else {
		fprintf(stderr,
		        "tape narrow backward: unrecognised shape pair "
		        "(a rank=%d shape=[%d,%d], r rank=%d shape=[%d,%d]).\n",
		        a->rank, a->rank > 0 ? a->shape[0] : 0, a->rank > 1 ? a->shape[1] : 0, r->rank,
		        r->rank > 0 ? r->shape[0] : 0, r->rank > 1 ? r->shape[1] : 0);
		abort();
	}
}

TAPE_REGISTER_OP(OP_NARROW, tape_backward_narrow)

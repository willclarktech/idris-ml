/* backend_tape/tape.c — TypedArena<TapeEntry> machinery + tape_append/
 * tape_reset + no_grad mechanism.
 *
 * Standalone TU compiled into backend_tape_tape.o.
 */

#include <stdlib.h>
#include <string.h>
#include "tape.h"
#include "arena.h"  /* arena_reset (called from tape_reset) */
#include "tensor.h" /* Tensor struct (used via tape entries' result/arg fields) */

/* ----------------------------------------------------------------
 * Deterministic BLAS — pin Accelerate (vecLib) to a single thread.
 *
 * Multi-threaded Accelerate GEMM/GEMV combine partial sums in a
 * work-stealing (nondeterministic) order, so two same-seed tape runs
 * drift bit-for-bit after enough steps — turning the ">=5-seed
 * convergence" methodology unsound (an `ntm-copy` seed flipped pass
 * <-> fail across reruns; root-caused 2026-06-18 — VECLIB_MAXIMUM_-
 * THREADS=1 made 8/8 runs bit-identical). Pin to 1 thread so tape —
 * the lean CPU *reference/correctness* backend — is reproducible by
 * construction. idris-ml's matrices are small (Apple AMX handles them
 * single-threaded), so the perf cost is negligible.
 *
 * `overwrite=0`: a user who wants multi-threaded BLAS over
 * reproducibility can still export VECLIB_MAXIMUM_THREADS=N. The
 * constructor runs at dlopen of libidrisml — before the first BLAS
 * call — so Accelerate reads the pinned value on its lazy init. */
__attribute__((constructor)) static void tape_pin_blas_single_thread(void) {
	setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
}

/* ----------------------------------------------------------------
 * TypedArena<T> — fixed-element-size linked-list arena. Struct is
 * declared in tape.h so backward / profiling can read tape_size
 * via the macro without an accessor function call.
 * ---------------------------------------------------------------- */

static void* typed_arena_append(TypedArena* a) {
	if (!a->head) {
		a->head = malloc(sizeof(TypedArenaChunk));
		a->head->data = calloc(a->chunk_capacity, a->element_size);
		a->head->next = NULL;
		a->tail = a->head;
		a->tail_count = 0;
	} else if (a->tail_count == a->chunk_capacity) {
		if (a->tail->next) {
			a->tail = a->tail->next;
		} else {
			TypedArenaChunk* c = malloc(sizeof(TypedArenaChunk));
			c->data = calloc(a->chunk_capacity, a->element_size);
			c->next = NULL;
			a->tail->next = c;
			a->tail = c;
		}
		a->tail_count = 0;
	}
	void* p = (char*)a->tail->data + a->tail_count * a->element_size;
	a->size++;
	a->tail_count++;
	return p;
}

static void typed_arena_reset(TypedArena* a) {
	a->size = 0;
	a->tail = a->head;
	a->tail_count = 0;
}

/* ----------------------------------------------------------------
 * Tape state. The 64K-entry chunk capacity gives ~5 MB at
 * sizeof(TapeEntry) ~ 80 bytes — large enough to hold one epoch of
 * a transformer-sized workload without ever growing.
 * ---------------------------------------------------------------- */

#define TAPE_CHUNK_SIZE (1 << 16)

TypedArena tape_arena = {
    .head = NULL,
    .tail = NULL,
    .size = 0,
    .tail_count = 0,
    .chunk_capacity = TAPE_CHUNK_SIZE,
    .element_size = sizeof(TapeEntry),
};

long g_tape_peak = 0;

/* Forward declaration: _wall_ms is defined in the profiling section of
 * backend_tape.c (still monolithic for now). The profiling globals it
 * touches are also defined there. */
extern double _wall_ms(void);
extern double prof_forward_per_op[];
extern int prof_forward_count_per_op[];
extern double prof_op_t_prev;

/* When > 0, tape_append is a no-op and any tensor created inside is
   marked requires_grad=0. Used by withNoGrad: rollouts, evals, any
   forward that doesn't need gradients. Counter (not bool) so nested
   withNoGrad scopes nest correctly. Mirrors PyTorch's torch.no_grad().
   The previous tensor_no_grad_begin/end were stubs; now wired up. */
int no_grad_depth = 0;

/* Dummy tape entry returned by tape_append in no_grad mode. Many
   callers do `e = tape_append(...); e->op_meta = ...;` — they need
   a valid (non-null) pointer to write to. We give them this static
   buffer; the writes are scratch and never read (the result tensor
   has tape_idx=-1 so backward never reaches it). */
static TapeEntry _no_grad_dummy_entry;

TapeEntry* tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
	/* Inside withNoGrad: skip the tape entry entirely and mark the
	   result as not grad-tracked, so downstream ops don't propagate
	   grad through it. Saves both forward overhead (no entry) and
	   backward overhead (fewer entries to walk). */
	if (no_grad_depth > 0) {
		if (result) {
			result->requires_grad = 0;
			result->tape_idx = -1;
		}
		/* Return a writable dummy so callers that do
		      e = tape_append(...); e->op_meta = ...;
		   don't crash. The result has tape_idx=-1 so backward never
		   dereferences this entry — the writes are scratch. */
		memset(&_no_grad_dummy_entry, 0, sizeof(_no_grad_dummy_entry));
		return &_no_grad_dummy_entry;
	}
	/* Attribute the time since the previous tape_append (or epoch_begin)
	   to this op — covers its compute + setup. Idris-side glue between
	   ops adds small noise, but at typical 30+ µs per op the signal
	   still surfaces a 2× regression cleanly. */
	if (prof_op_t_prev > 0 && op >= 0 && op < OP_COUNT) {
		double now = _wall_ms();
		prof_forward_per_op[op] += now - prof_op_t_prev;
		prof_forward_count_per_op[op]++;
		prof_op_t_prev = now;
	}
	TapeEntry* e = (TapeEntry*)typed_arena_append(&tape_arena);
	if ((long)tape_size > g_tape_peak) g_tape_peak = (long)tape_size;
	memset(e, 0, sizeof(TapeEntry));
	e->op = op;
	e->result = result;
	e->arg1 = arg1;
	e->arg2 = arg2;
	e->scalar_arg = scalar_arg;
	if (result) result->tape_idx = tape_arena.size - 1;
	return e;
}

void tape_reset(void) {
	/* Walk chunks tearing down per-entry heap allocations before reset.
	   Each entry's op_meta and inputs were heap-allocated by the forward
	   op and must be freed before the entry is reused on the next epoch. */
	int remaining = tape_arena.size;
	for (TypedArenaChunk* c = tape_arena.head; c && remaining > 0; c = c->next) {
		int n = remaining > tape_arena.chunk_capacity ? tape_arena.chunk_capacity : remaining;
		TapeEntry* entries = (TapeEntry*)c->data;
		for (int i = 0; i < n; i++) {
			TapeEntry* e = &entries[i];
			/* Free OP_STACK inputs arrays */
			if (e->op == OP_STACK && e->inputs) {
				free((void*)e->inputs);
				e->inputs = NULL;
			}
			/* Free OP_LAYER_NORM_2D heap arrays */
			if (e->op == OP_LAYER_NORM_2D && e->op_meta) {
				LayerNormMeta* meta = (LayerNormMeta*)e->op_meta;
				free(meta->x_hat);
				free(meta->rstd);
				meta->x_hat = NULL;
				meta->rstd = NULL;
			}
			/* Free OP_RMS_NORM_2D heap arrays */
			if (e->op == OP_RMS_NORM_2D && e->op_meta) {
				RmsNormMeta* meta = (RmsNormMeta*)e->op_meta;
				free(meta->x_hat);
				free(meta->rstd);
				meta->x_hat = NULL;
				meta->rstd = NULL;
			}
			/* Free OP_SWIGLU_2D heap arrays */
			if (e->op == OP_SWIGLU_2D && e->op_meta) {
				SwiGluMeta* meta = (SwiGluMeta*)e->op_meta;
				free(meta->sig_g);
				meta->sig_g = NULL;
			}
			/* Free OP_GRU_CELL gate arrays. `prev` is a Tensor* owned by
			   the caller (typically a per-sequence state handle); we don't
			   own it. */
			if (e->op == OP_GRU_CELL && e->op_meta) {
				GruCellMeta* meta = (GruCellMeta*)e->op_meta;
				free(meta->zG);
				free(meta->rG);
				free(meta->nG);
				meta->zG = meta->rG = meta->nG = NULL;
				meta->prev = NULL;
			}
			/* Free OP_EMBEDDING indices */
			if (e->op == OP_EMBEDDING && e->op_meta) {
				EmbeddingMeta* meta = (EmbeddingMeta*)e->op_meta;
				free(meta->indices);
				meta->indices = NULL;
			}
			/* Free OP_BATCH_NORM arrays */
			if (e->op == OP_BATCH_NORM && e->op_meta) {
				BatchNormMeta* meta = (BatchNormMeta*)e->op_meta;
				free(meta->x_hat);
				free(meta->rstd);
				meta->x_hat = NULL;
				meta->rstd = NULL;
			}
			/* Free OP_DROPOUT mask */
			if (e->op == OP_DROPOUT && e->op_meta) {
				DropoutMeta* meta = (DropoutMeta*)e->op_meta;
				free(meta->mask);
				meta->mask = NULL;
			}
			/* Free OP_MAX_POOL1D max indices */
			if (e->op == OP_MAX_POOL1D && e->op_meta) {
				MaxPool1DMeta* meta = (MaxPool1DMeta*)e->op_meta;
				free(meta->max_indices);
				meta->max_indices = NULL;
			}
			/* Free OP_MAX_POOL2D max indices */
			if (e->op == OP_MAX_POOL2D && e->op_meta) {
				MaxPool2DMeta* meta = (MaxPool2DMeta*)e->op_meta;
				free(meta->max_indices);
				meta->max_indices = NULL;
			}
			/* Free OP_MAX_POOL2D_BATCHED max indices */
			if (e->op == OP_MAX_POOL2D_BATCHED && e->op_meta) {
				MaxPool2DBatchedMeta* meta = (MaxPool2DBatchedMeta*)e->op_meta;
				free(meta->max_indices);
				meta->max_indices = NULL;
			}
			/* Free grad arrays on non-persistent (arena) tensors.
			   These are heap-allocated by ensure_grad during backward. */
			Tensor* r = e->result;
			if (r && !r->persistent && r->grad) {
				free(r->grad);
				r->grad = NULL;
			}
		}
		remaining -= n;
	}
	typed_arena_reset(&tape_arena);
	arena_reset();
}

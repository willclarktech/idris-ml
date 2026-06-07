/* training/autograd/backward.c — tape-walk backward driver.
 *
 * The single entry point that runs the reverse-mode
 * autodiff loop. Every per-op backward function is registered via
 * TAPE_REGISTER_OP at load time; this loop walks the tape in reverse
 * (via the chunked TypedArena) and calls the dispatch-table entry
 * for each. OP_CONST tape entries fall through (no backward).
 *
 * Profiling globals (prof_forward_ms, prof_backward_*, prof_epoch_start,
 * prof_op_t_prev) live in backend_tape.c for now; a subsequent lift
 * will move them to a dedicated profiling.c.
 */

#include <stdlib.h>
#include <sys/time.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "op_dispatch.h"
#include "../../../backend.h"

/* From the still-monolithic profiling cluster in backend_tape.c — these
   extern decls go away when profiling lifts to its own TU. */
extern double _wall_ms(void);
extern double prof_forward_ms, prof_backward_ms;
extern int prof_backward_ops, prof_backward_processed, prof_backward_skipped;
extern double prof_epoch_start;
extern double prof_op_t_prev;
extern double prof_backward_per_op[OP_COUNT];
extern int prof_backward_count_per_op[OP_COUNT];

/* Diagnostic dump — defined in backend_tape.c alongside param_registry. */
extern void _dbg_dump_param_grads_if_enabled(void);

void tensor_backward(TensorHandle h) {
	double t0 = _wall_ms();
	/* Attribute time since epoch_begin to forward */
	if (prof_epoch_start > 0) {
		prof_forward_ms += t0 - prof_epoch_start;
		prof_epoch_start = 0;
	}
	/* Stop per-op forward accumulation; the next epoch_begin will rearm. */
	prof_op_t_prev = 0;
	Tensor* loss = (Tensor*)h;
	if (loss->tape_idx < 0) return;

	/* Initialize loss gradient to 1.0. Typed allocation matches the
	   loss's data dtype so the grad buffer is F32-sized for F32 losses
	   (required by the Row 38 symmetric-grad path — mismatched
	   allocation + typed accumulators would read the high bytes of
	   the F64 1.0 encoding as garbage on F32 reads). */
	ensure_grad(loss);
	tape_grad_store_d(loss, 0, 1.0);

	int processed = 0, skipped = 0;

	/* Walk tape in reverse via chunk-array — same semantics as the old
	   `for (int i = loss->tape_idx; i >= 0; i--) { TapeEntry* e = &tape[i]; }`
	   but indexes the chunked tape directly so the cost stays O(N) total. */
	int _num_chunks_b = 0;
	for (TypedArenaChunk* _c = tape_arena.head; _c; _c = _c->next)
		_num_chunks_b++;
	/* `loss->tape_idx >= 0` (checked above) implies at least one chunk; the
	 * explicit guard here is for the static analyzer, which can't prove the
	 * connection and otherwise flags the malloc-of-size-zero path. */
	if (_num_chunks_b == 0) return;
	TypedArenaChunk** _chunks_b =
	    (TypedArenaChunk**)malloc(_num_chunks_b * sizeof(TypedArenaChunk*));
	{
		int _ci = 0;
		for (TypedArenaChunk* _c = tape_arena.head; _c; _c = _c->next)
			_chunks_b[_ci++] = _c;
	}
	int _start_cidx = loss->tape_idx / TAPE_CHUNK_SIZE;
	int _start_intra = loss->tape_idx % TAPE_CHUNK_SIZE;
	for (int _cidx = _start_cidx; _cidx >= 0; _cidx--) {
		TapeEntry* _entries_b = (TapeEntry*)_chunks_b[_cidx]->data;
		int _last_intra = (_cidx == _start_cidx) ? _start_intra : TAPE_CHUNK_SIZE - 1;
		for (int _j = _last_intra; _j >= 0; _j--) {
			TapeEntry* e = &_entries_b[_j];
			Tensor* r = e->result;
			if (!r->grad) {
				skipped++;
				continue;
			}
			processed++;
			double t_op = _wall_ms();

			/* Every OP_* now resolves through the dispatch table (populated
			   by each op's TAPE_REGISTER_OP at load time). OP_CONST is the
			   one exception: a leaf marker with no backward semantics — its
			   tape entries fall straight through the `if (!_fn)` skip. */
			TapeBackwardFn _fn = tape_dispatch_get(e->op);
			if (_fn) _fn(e);
			/* Accumulate per-op timing */
			if (e->op < OP_COUNT) {
				prof_backward_per_op[e->op] += _wall_ms() - t_op;
				prof_backward_count_per_op[e->op]++;
			}
		} /* close inner _j loop */
	} /* close outer _cidx loop */
	free((void*)_chunks_b);
	prof_backward_processed += processed;
	prof_backward_skipped += skipped;
	prof_backward_ms += _wall_ms() - t0;
	prof_backward_ops += processed;

	/* When DEBUG_PARAM_GRADS is set, dump per-param gradient L2 norm
	   to stderr. Use to identify zero/NaN/wrong-magnitude grads after
	   a single backward pass. */
	_dbg_dump_param_grads_if_enabled();
}

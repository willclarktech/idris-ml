/* backend_tape/training/autograd/op_dispatch.c — per-op backward
 * function dispatch table.
 *
 * Standalone TU.
 */

#include <stdio.h>
#include <stdlib.h>
#include "op_dispatch.h"
#include "../../tape.h" /* OP_COUNT enum (transitive via op_dispatch.h, but direct per include-cleaner) */

static TapeBackwardFn g_tape_backward[OP_COUNT] = {0};

void tape_register_op(int op, TapeBackwardFn fn) {
	if (op < 0 || op >= OP_COUNT) {
		// GCOVR_EXCL_START — death-tested by Test(op_dispatch, register_out_of_range_aborts); body
		// never returns
		fprintf(stderr,
		        "[tape backend] tape_register_op: op tag %d out of range "
		        "[0..%d)\n",
		        op, OP_COUNT);
		// NOLINTNEXTLINE(misc-include-cleaner): macOS SDK: abort via _abort.h umbrella
		abort();
		// GCOVR_EXCL_STOP
	}
	g_tape_backward[op] = fn;
}

TapeBackwardFn tape_dispatch_get(int op) {
	if (op < 0 || op >= OP_COUNT) return NULL;
	return g_tape_backward[op];
}

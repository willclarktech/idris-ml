/* backend_tape/training/autograd/op_dispatch.c — per-op backward
 * function dispatch table.
 *
 * Phase 1.0.3 (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 *
 * The table is BSS-zero by default — any op tag with no registered
 * handler reads as NULL. Constructor-time registration (via
 * TAPE_REGISTER_OP in each op file) populates entries before
 * tensor_backward is ever called from Idris.
 *
 * Currently #included from backend_tape.c (single-TU build); Phase
 * 1.0.4 will compile this as its own TU.
 *
 * Symbols are deliberately non-static so the Criterion test suite
 * (packages/backends/test/tape/training/autograd/test_op_dispatch.c)
 * can verify the BSS-zero invariant + register/get round-trip.
 * They are NOT in backend.h — the rename header won't suffix them.
 */

static TapeBackwardFn g_tape_backward[OP_COUNT] = {0};

void tape_register_op(int op, TapeBackwardFn fn) {
    if (op < 0 || op >= OP_COUNT) {
        fprintf(stderr, "[tape backend] tape_register_op: op tag %d out of range "
                "[0..%d)\n", op, OP_COUNT);
        abort();
    }
    g_tape_backward[op] = fn;
}

TapeBackwardFn tape_dispatch_get(int op) {
    if (op < 0 || op >= OP_COUNT) return NULL;
    return g_tape_backward[op];
}

/* backend_tape/training/autograd/op_dispatch.h — per-op backward
 * function dispatch table.
 *
 * The table was initially unused — the monolithic `tensor_backward()`
 * switch was retained while the per-op migration ran. The migration
 * uses `TAPE_REGISTER_OP(op, fn)` (defined here) inside each op's
 * source file to populate `g_tape_backward[op]` at load time via a
 * constructor. Once every op-tag had a registered handler, the
 * dispatch-table cutover flipped `tensor_backward()` to dispatch
 * through the table and the 1848-line switch disappeared.
 *
 * Originally #included from backend_tape.c (single-TU build); the
 * per-TU compile rule makes it a public header.
 */

#ifndef IDRISML_BACKEND_TAPE_OP_DISPATCH_H
#define IDRISML_BACKEND_TAPE_OP_DISPATCH_H

#include "../../tape.h"  /* OP_COUNT, TapeEntry */

/* Backward kernel signature: receives the forward's TapeEntry; reads
 * inputs through tape_load_d and writes to .grad. */
typedef void (*TapeBackwardFn)(TapeEntry*);

/* Register a backward function for op tag `op`. Idempotent: subsequent
 * calls overwrite. Bounds-checked at registration; out-of-range `op`
 * aborts loudly. */
void tape_register_op(int op, TapeBackwardFn fn);

/* Get the registered backward function for op tag `op`, or NULL if
 * none has been registered. */
TapeBackwardFn tape_dispatch_get(int op);

/* TAPE_REGISTER_OP(op, fn) — define a __attribute__((constructor)) that
 * fires at load time and registers `fn` as op's backward handler. Each
 * op's source file places one of these at file scope:
 *
 *     TAPE_REGISTER_OP(OP_ADD, tape_backward_add);
 *
 * Constructor order across TUs is unspecified, but the dispatch table
 * is a BSS-zero global so reads before any constructor fires return
 * NULL (the well-defined unregistered state). No reader runs before
 * Idris invokes tensor_backward, by which time all constructors have
 * completed (dyld's standard guarantee).
 */
#define TAPE_REGISTER_OP(op, fn)                                       \
    __attribute__((constructor))                                       \
    static void _tape_reg_##op##_##fn(void) { tape_register_op(op, fn); }

#endif /* IDRISML_BACKEND_TAPE_OP_DISPATCH_H */

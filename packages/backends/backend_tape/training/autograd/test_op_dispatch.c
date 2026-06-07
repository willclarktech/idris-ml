/* Criterion suite for the tape op-dispatch table.
 *
 * Verifies the dispatch-table mechanism that the per-op TAPE_REGISTER_OP
 * constructors populate:
 *   - BSS-zero invariant for the permanently-unregistered slot OP_CONST
 *     (leaf marker — no backward function expected).
 *   - tape_register_op / tape_dispatch_get round-trip.
 *   - Out-of-range queries return NULL (no abort, no UB).
 *
 * The closeout pass (after every OP_* migrated to its own file) flipped
 * this from "test unmigrated slots stay NULL" to "test OP_CONST stays
 * NULL" — OP_CONST is the only op whose backward semantics are "no-op"
 * by design, so its slot is permanently NULL even after full migration.
 */

#include <criterion/criterion.h>
#include "backend_tape/tape.h"
#include "backend_tape/training/autograd/op_dispatch.h"

#ifdef BACKEND_TAPE

static void noop_backward(TapeEntry* e) {
	(void)e;
}
static void other_backward(TapeEntry* e) {
	(void)e;
}

Test(op_dispatch, op_const_is_null) {
	/* OP_CONST is the leaf-tensor marker — its tape entries skip backward
	   entirely (the explicit `case OP_CONST: break;` in tensor_backward
	   was always a no-op). Dispatch table mirrors that: slot stays NULL. */
	cr_assert_null(tape_dispatch_get(OP_CONST),
	               "g_tape_backward[OP_CONST] should be NULL — leaf marker has no backward");
}

Test(op_dispatch, migrated_ops_are_populated) {
	/* Catch a constructor not firing (e.g., the .o accidentally missing
	   from the dylib link). Sample a representative migrated op. */
	cr_assert_not_null(tape_dispatch_get(OP_ADD),
	                   "g_tape_backward[OP_ADD] should be populated by core/elementwise/add.c");
}

Test(op_dispatch, register_then_get_roundtrip) {
	/* Register a stub on OP_CONST (permanently NULL) and verify
	   retrieval. Each Criterion Test() runs in its own forked process,
	   so writes here do not leak to siblings — Criterion's process
	   isolation provides the cleanup the explicit reset at the end of
	   this test is a redundant belt-and-braces guard for. */
	TapeBackwardFn original = tape_dispatch_get(OP_CONST);
	cr_assert_null(original);
	tape_register_op(OP_CONST, noop_backward);
	cr_assert_eq(tape_dispatch_get(OP_CONST), noop_backward,
	             "tape_register_op + tape_dispatch_get should round-trip");
	/* Overwrite — register is idempotent (last-write-wins). */
	tape_register_op(OP_CONST, other_backward);
	cr_assert_eq(tape_dispatch_get(OP_CONST), other_backward,
	             "second tape_register_op should overwrite");
	/* Cleanup (belt-and-braces; Criterion fork isolates anyway) */
	tape_register_op(OP_CONST, original);
}

Test(op_dispatch, out_of_range_returns_null) {
	/* Negative and >= OP_COUNT queries must return NULL, not crash. */
	cr_assert_null(tape_dispatch_get(-1));
	cr_assert_null(tape_dispatch_get(OP_COUNT));
	cr_assert_null(tape_dispatch_get(OP_COUNT + 1000));
}

#endif /* BACKEND_TAPE */

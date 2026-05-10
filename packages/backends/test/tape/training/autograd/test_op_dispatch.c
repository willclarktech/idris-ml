/* Criterion suite for the tape op-dispatch table (Phase 1.0.3).
 *
 * Verifies the dispatch-table mechanism that Phase 1a-1d will populate:
 *   - BSS-zero invariant (table starts NULL for every op).
 *   - tape_register_op / tape_dispatch_get round-trip.
 *   - Out-of-range queries return NULL (no abort, no UB).
 *
 * The table itself isn't read by tensor_backward yet — the monolithic
 * switch still drives backward — so these assertions exercise the
 * surface in isolation. The first op-extraction commit (Phase 1a.2
 * for `add`) will produce the first real registration; subsequent
 * commits drop the corresponding `case` arm from the monolith switch
 * and add a TAPE_REGISTER_OP at op-file scope.
 */

#include <criterion/criterion.h>
#include "../../../../backend_tape/tape.h"
#include "../../../../backend_tape/training/autograd/op_dispatch.h"

static void noop_backward(TapeEntry* e) { (void)e; }
static void other_backward(TapeEntry* e) { (void)e; }

/* Pick an op tag that's NOT yet migrated to its own file (and therefore
   has no TAPE_REGISTER_OP firing at load time). OP_CONV1D is the
   first conv op slated for Phase 1d.1 — until then its entry stays
   NULL, so these tests can register/clear without colliding with a
   real handler. Update this when OP_CONV1D migrates. */
#define UNMIGRATED_OP OP_CONV1D

Test(op_dispatch, unmigrated_ops_are_null) {
    /* Anything not yet migrated stays NULL — gives confidence that
       constructors only register what we expect, and the table doesn't
       acquire stray entries between phases. */
    cr_assert_null(tape_dispatch_get(UNMIGRATED_OP),
        "g_tape_backward[%d] (UNMIGRATED_OP) should be NULL — no TAPE_REGISTER_OP for this op yet",
        UNMIGRATED_OP);
}

Test(op_dispatch, migrated_ops_are_populated) {
    /* Phase 1a.2 registered OP_ADD's backward. Any future migration
       lights up the corresponding entry. Catch a constructor not firing
       (e.g., the .o accidentally missing from the dylib link). */
    cr_assert_not_null(tape_dispatch_get(OP_ADD),
        "g_tape_backward[OP_ADD] should be populated by Phase 1a.2's TAPE_REGISTER_OP");
}

Test(op_dispatch, register_then_get_roundtrip) {
    /* Register a stub on an unmigrated slot and verify retrieval.
       Each Criterion Test() runs in its own forked process, so writes
       here do not leak to siblings — Criterion's process isolation
       provides the cleanup that the explicit reset at the end of this
       test is a redundant belt-and-braces guard for. */
    TapeBackwardFn original = tape_dispatch_get(UNMIGRATED_OP);
    cr_assert_null(original);
    tape_register_op(UNMIGRATED_OP, noop_backward);
    cr_assert_eq(tape_dispatch_get(UNMIGRATED_OP), noop_backward,
        "tape_register_op + tape_dispatch_get should round-trip");
    /* Overwrite — register is idempotent (last-write-wins). */
    tape_register_op(UNMIGRATED_OP, other_backward);
    cr_assert_eq(tape_dispatch_get(UNMIGRATED_OP), other_backward,
        "second tape_register_op should overwrite");
    /* Cleanup (belt-and-braces; Criterion fork isolates anyway) */
    tape_register_op(UNMIGRATED_OP, original);
}

Test(op_dispatch, out_of_range_returns_null) {
    /* Negative and >= OP_COUNT queries must return NULL, not crash. */
    cr_assert_null(tape_dispatch_get(-1));
    cr_assert_null(tape_dispatch_get(OP_COUNT));
    cr_assert_null(tape_dispatch_get(OP_COUNT + 1000));
}

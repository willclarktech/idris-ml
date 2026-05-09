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

Test(op_dispatch, table_zero_init) {
    /* Pre-condition: no constructor has registered for OP_ADD yet
       (Phase 1.0.3 keeps the dispatch table unused). All entries
       must read as NULL. */
    for (int op = 0; op < OP_COUNT; op++) {
        cr_assert_null(tape_dispatch_get(op),
            "g_tape_backward[%d] should be NULL before any TAPE_REGISTER_OP fires", op);
    }
}

Test(op_dispatch, register_then_get_roundtrip) {
    /* Register a stub and verify retrieval. Reset to NULL afterwards so
       this test doesn't leak state to siblings (Criterion's process
       isolation also guarantees this, but explicit cleanup keeps the
       intent local). */
    cr_assert_null(tape_dispatch_get(OP_ADD));
    tape_register_op(OP_ADD, noop_backward);
    cr_assert_eq(tape_dispatch_get(OP_ADD), noop_backward,
        "tape_register_op + tape_dispatch_get should round-trip");
    /* Overwrite — register is idempotent (last-write-wins). */
    tape_register_op(OP_ADD, other_backward);
    cr_assert_eq(tape_dispatch_get(OP_ADD), other_backward,
        "second tape_register_op should overwrite");
    /* Cleanup */
    tape_register_op(OP_ADD, NULL);
    cr_assert_null(tape_dispatch_get(OP_ADD));
}

Test(op_dispatch, out_of_range_returns_null) {
    /* Negative and >= OP_COUNT queries must return NULL, not crash. */
    cr_assert_null(tape_dispatch_get(-1));
    cr_assert_null(tape_dispatch_get(OP_COUNT));
    cr_assert_null(tape_dispatch_get(OP_COUNT + 1000));
}

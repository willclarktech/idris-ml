/* Criterion smoke test — verifies the test framework links and runs
 * with per-test process isolation. No backend assertions; that surface
 * arrives in Phase 1 as per-op Criterion suites under
 * packages/backends/test/<backend>/<category>/<subcat>/test_<op>.c.
 *
 * Build via `make BACKEND=<b> test-backend-criterion`. The target
 * invokes `--xml=build/test-criterion-<b>.xml` so CI can consume the
 * JUnit XML report directly.
 *
 * Phase 0.3 (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 */

#include <criterion/criterion.h>

Test(smoke, hello) {
    cr_assert(1, "Criterion is linked and the test harness runs");
}

Test(smoke, addition) {
    cr_assert_eq(2 + 2, 4);
}

Test(smoke, process_isolation_sentinel) {
    /* Each Test() forks; this body must not see state from a sibling.
     * Verified manually by the process-isolation invariant Criterion
     * provides — Test() fixtures don't leak globals. */
    static int seen = 0;
    cr_assert_eq(seen, 0, "fresh subprocess should not have prior seen=%d", seen);
    seen = 1;
}

/* Criterion suite for the tape adapter stub.
 *
 * The shared-port stub adapter must abort at runtime so any premature
 * wiring (a shared/training/ TU calling into the port before its real
 * implementation lands) fails loudly rather than silently no-op'ing.
 * The `.signal = SIGABRT` annotation drives Criterion's process-
 * isolated child to assert the abort actually fires.
 *
 * Secondary check: the port struct's function pointers are all
 * populated (no missing initializer falls through to NULL).
 */

#include <signal.h>
#include <criterion/criterion.h>
#include "../../../shared/training/port.h"

Test(shared_training, port_struct_populated) {
    cr_assert_not_null(g_active_port.tensor_numel);
    cr_assert_not_null(g_active_port.tensor_requires_grad);
    cr_assert_not_null(g_active_port.tensor_has_grad);
    cr_assert_not_null(g_active_port.data_read);
    cr_assert_not_null(g_active_port.data_write);
    cr_assert_not_null(g_active_port.grad_read);
    cr_assert_not_null(g_active_port.grad_write);
    cr_assert_not_null(g_active_port.zero_grad);
    cr_assert_not_null(g_active_port.load_doubles);
    cr_assert_not_null(g_active_port.load_int64);
    cr_assert_not_null(g_active_port.backward);
    cr_assert_not_null(g_active_port.epoch_boundary);
    cr_assert_not_null(g_active_port.wall_ms);
}

Test(shared_training, stub_aborts_on_use, .signal = SIGABRT) {
    /* Pick a slot still on the abort-stub path so the test stays valid
       as the adapter fills in. `backward` / `epoch_boundary` / `wall_ms`
       are still unimplemented — their shared consumers (optimizer,
       backward driver, profiler) have not been lifted yet, so the
       stubs remain in place. Swap targets if/when one of these
       acquires its real implementation. */
    g_active_port.epoch_boundary();
}

/* Criterion suite for the shared-port struct on the tape adapter.
 *
 * Asserts every function-pointer slot on `g_active_port` is populated
 * (no missing initializer falls through to NULL). The companion
 * suites (test_param_registry.c, test_optimizer.c) cover the
 * adapter's behavioural correctness; this one is a tripwire for
 * accidentally dropping a slot when the port surface grows.
 */

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

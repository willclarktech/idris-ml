/* Criterion suite for the shared-port struct on the tape adapter.
 *
 * Asserts every function-pointer slot on `g_active_port` is populated
 * (no missing initializer falls through to NULL). The companion
 * suites (test_param_registry.c, test_optimizer.c) cover the
 * adapter's behavioural correctness; this one is a tripwire for
 * accidentally dropping a slot when the port surface grows.
 */

#include <criterion/criterion.h>
#include "shared/training/port.h"

#ifdef BACKEND_TAPE

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
    cr_assert_not_null(g_active_port.wall_ms);
    cr_assert_not_null(g_active_port.create_scalar);
    cr_assert_not_null(g_active_port.create);
    cr_assert_not_null(g_active_port.create_1d);
    cr_assert_not_null(g_active_port.create_2d);
    cr_assert_not_null(g_active_port.create_param_1d);
    cr_assert_not_null(g_active_port.create_param_2d);
    cr_assert_not_null(g_active_port.create_param_3d);
    cr_assert_not_null(g_active_port.create_param_4d);
    cr_assert_not_null(g_active_port.create_state_1d);
    cr_assert_not_null(g_active_port.create_state_2d);
    cr_assert_not_null(g_active_port.cast_dtype);
    cr_assert_not_null(g_active_port.optimizer_create_sgd);
    cr_assert_not_null(g_active_port.optimizer_create_rmsprop);
    cr_assert_not_null(g_active_port.optimizer_create_adam);
    cr_assert_not_null(g_active_port.optimizer_create_adam_group);
    cr_assert_not_null(g_active_port.optimizer_create_adamw);
    cr_assert_not_null(g_active_port.optimizer_free);
    cr_assert_not_null(g_active_port.optimizer_set_lr);
    cr_assert_not_null(g_active_port.optimizer_set_param_lr);
    cr_assert_not_null(g_active_port.optimizer_step);
    cr_assert_not_null(g_active_port.optimizer_clip_grad_value_filtered);
    cr_assert_not_null(g_active_port.optimizer_clip_grad_norm_filtered);
    cr_assert_not_null(g_active_port.optimizer_buf_count);
    cr_assert_not_null(g_active_port.optimizer_get_m);
    cr_assert_not_null(g_active_port.optimizer_get_v);
    cr_assert_not_null(g_active_port.optimizer_set_m);
    cr_assert_not_null(g_active_port.optimizer_set_v);
    cr_assert_not_null(g_active_port.optimizer_get_meta);
    cr_assert_not_null(g_active_port.optimizer_set_meta);
}

#endif /* BACKEND_TAPE */

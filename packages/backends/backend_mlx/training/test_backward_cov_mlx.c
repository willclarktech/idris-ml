/* mlx-only Criterion suite — tensor_backward early-return arm (backward.cpp).
 *
 * A tensor created without requires_grad is not tape-tracked (tape_idx < 0).
 * Calling tensor_backward on it must take the early-return arm (no tape walk).
 * No existing mlx test drives backward on an untracked tensor.
 */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

Test(mlx_backward_cov, backward_on_constant_is_noop) {
	param_clear();
	TensorHandle c = tensor_create_scalar(3.0, /*requires_grad=*/0); /* tape_idx < 0 */
	tensor_backward(c); /* exercises the tape_idx < 0 early-return */
	cr_assert_float_eq(tensor_item(c), 3.0, TEST_TOL_TIGHT, "value unchanged (got %.6f)",
	                   tensor_item(c));
	param_clear();
}

#endif /* BACKEND_MLX */

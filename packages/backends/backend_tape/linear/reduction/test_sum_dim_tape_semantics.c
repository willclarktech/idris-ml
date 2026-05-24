/* Criterion suite for tape-specific `tensor_sum_dim` semantics.
 *
 * Lives under test/tape/ (not test/common/) because the assertion only
 * holds on tape: tape's `tensor_sum_dim` is a simplified-semantics
 * shim that delegates to `tensor_sum`, returning a scalar. Torch and
 * mlx implement proper sum-along-dim and return a rank-reduced tensor,
 * so `tensor_item(r)` on them would assert (Tensor with >1 elements
 * → Scalar conversion error).
 */

#include <criterion/criterion.h>
#include "backend.h"

#ifdef BACKEND_TAPE

Test(tape_linear_reduction_sum_dim, simplified_delegates_to_sum) {
    double d[] = {1.0, 2.0, 3.0, 4.0};
    int s[] = {2, 2};
    TensorHandle t = tensor_create(d, s, 2, 0);
    TensorHandle r = tensor_sum_dim(t, 0, 0);
    cr_assert_float_eq(tensor_item(r), 10.0, 1e-12);
}

#endif /* BACKEND_TAPE */

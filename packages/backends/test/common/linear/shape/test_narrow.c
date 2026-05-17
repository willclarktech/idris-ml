/* Criterion suite for `tensor_narrow`.
 *
 * Backend reality at 2026-05-26: tape supports rank=2 dim=1 natively
 * (see `backend_tape/linear/shape/narrow.c`); torch and mlx kernels
 * historically ignored the `dim` argument (`(void)dim;`) and flatten
 * the input before narrowing axis-0. That meant `primNarrow ... 1 ...`
 * was a silent shape lie on those backends — BERT's multi-head Q/K/V
 * narrow was producing garbage tensors that happened to land within
 * the 1e-3 oracle tolerance.
 *
 * `axis1_correctness_rank2` is the cross-backend gate that pins the
 * intended semantics. Tape passes today; torch and mlx must be fixed
 * to pass.
 */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(linear_shape_narrow, forward_slice) {
    double d[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    int s[] = {5};
    TensorHandle v = tensor_create(d, s, 1, 0);
    TensorHandle n = tensor_narrow(v, 0, 1, 3);  /* [20, 30, 40] */
    double out[3];
    tensor_to_doubles(n, out);
    cr_assert_float_eq(out[0], 20.0, 1e-12);
    cr_assert_float_eq(out[1], 30.0, 1e-12);
    cr_assert_float_eq(out[2], 40.0, 1e-12);
}

Test(linear_shape_narrow, backward_scatters_to_offset) {
    /* narrow [v0..v4], 1..4 -> [v1, v2, v3]; sum -> backward should
       set parent's grad to [0, 1, 1, 1, 0]. */
    param_clear();
    double d[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    int s[] = {5};
    TensorHandle v = tensor_create(d, s, 1, 1);
    param_register("v", v);
    TensorHandle n = tensor_narrow(v, 0, 1, 3);
    TensorHandle loss = tensor_sum(n);
    tensor_backward(loss);
    cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "grad[0] should be 0");
    cr_assert_float_eq(param_grad_item_at(0, 1), 1.0, 1e-12,
        "grad[1] should be 1.0 (got %.6f)", param_grad_item_at(0, 1));
    cr_assert_float_eq(param_grad_item_at(0, 2), 1.0, 1e-12, "grad[2] should be 1.0");
    cr_assert_float_eq(param_grad_item_at(0, 3), 1.0, 1e-12, "grad[3] should be 1.0");
    cr_assert_float_eq(param_grad_item_at(0, 4), 0.0, 1e-12, "grad[4] should be 0");
}

Test(linear_shape_narrow, axis1_correctness_rank2) {
    /* [3, 6] input:
     *   0  1  2  3  4  5
     *   6  7  8  9 10 11
     *  12 13 14 15 16 17
     * narrow dim=1 start=2 length=3 must give [3, 3]:
     *   2  3  4
     *   8  9 10
     *  14 15 16
     */
    double d[] = {
        0.0,  1.0,  2.0,  3.0,  4.0,  5.0,
        6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
       12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    };
    int s[] = {3, 6};
    TensorHandle v = tensor_create(d, s, 2, 0);
    TensorHandle n = tensor_narrow(v, 1, 2, 3);

    /* Result must be rank-2, shape [3, 3]. */
    cr_assert_eq(tensor_dim(n), 2,
        "narrow rank should be 2 (got %d) — `dim` arg likely being ignored",
        tensor_dim(n));
    cr_assert_eq(tensor_size(n, 0), 3,
        "narrow shape[0] should be 3 (got %d)", tensor_size(n, 0));
    cr_assert_eq(tensor_size(n, 1), 3,
        "narrow shape[1] should be 3 (got %d)", tensor_size(n, 1));
    cr_assert_eq(tensor_numel(n), 9,
        "narrow numel should be 9 (got %d) — backend likely flattening before slice",
        tensor_numel(n));

    double out[9];
    tensor_to_doubles(n, out);
    cr_assert_float_eq(out[0],  2.0, 1e-12, "[0,0] should be 2.0  (got %.6f)", out[0]);
    cr_assert_float_eq(out[1],  3.0, 1e-12, "[0,1] should be 3.0  (got %.6f)", out[1]);
    cr_assert_float_eq(out[2],  4.0, 1e-12, "[0,2] should be 4.0  (got %.6f)", out[2]);
    cr_assert_float_eq(out[3],  8.0, 1e-12, "[1,0] should be 8.0  (got %.6f)", out[3]);
    cr_assert_float_eq(out[4],  9.0, 1e-12, "[1,1] should be 9.0  (got %.6f)", out[4]);
    cr_assert_float_eq(out[5], 10.0, 1e-12, "[1,2] should be 10.0 (got %.6f)", out[5]);
    cr_assert_float_eq(out[6], 14.0, 1e-12, "[2,0] should be 14.0 (got %.6f)", out[6]);
    cr_assert_float_eq(out[7], 15.0, 1e-12, "[2,1] should be 15.0 (got %.6f)", out[7]);
    cr_assert_float_eq(out[8], 16.0, 1e-12, "[2,2] should be 16.0 (got %.6f)", out[8]);
}

Test(linear_shape_narrow, axis1_backward_scatters_columns) {
    /* Sum-loss after axis=1 narrow: parent grad must scatter the
     * sliced columns and leave the rest at 0. */
    param_clear();
    double d[] = {
        0.0,  1.0,  2.0,  3.0,  4.0,  5.0,
        6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
       12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    };
    int s[] = {3, 6};
    TensorHandle v = tensor_create(d, s, 2, 1);
    param_register("v", v);
    TensorHandle n = tensor_narrow(v, 1, 2, 3);
    TensorHandle loss = tensor_sum(n);
    tensor_backward(loss);

    /* Expected grad: 1.0 at columns [2..4], 0.0 elsewhere — full [3, 6]. */
    double expected[18] = {
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    };
    for (int i = 0; i < 18; i++) {
        cr_assert_float_eq(param_grad_item_at(0, i), expected[i], 1e-12,
            "grad[%d] should be %.1f (got %.6f)",
            i, expected[i], param_grad_item_at(0, i));
    }
}

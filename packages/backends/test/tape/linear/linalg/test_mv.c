/* Criterion suite for tape `tensor_mv`. */

#include <criterion/criterion.h>
#include "../../../../backend.h"

Test(tape_linear_linalg_mv, forward_2x3_times_3vec) {
    /* M = [[1, 2, 3], [4, 5, 6]], v = [1, 1, 1]; M @ v = [6, 15] */
    double md[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double vd[] = {1.0, 1.0, 1.0};
    int sm[] = {2, 3};
    int sv[] = {3};
    TensorHandle mat = tensor_create(md, sm, 2, 0);
    TensorHandle vec = tensor_create(vd, sv, 1, 0);
    TensorHandle r = tensor_mv(mat, vec);
    cr_assert_eq(tensor_numel(r), 2);
    double out[2];
    tensor_to_doubles(r, out);
    cr_assert_float_eq(out[0],  6.0, 1e-12);
    cr_assert_float_eq(out[1], 15.0, 1e-12);
}

Test(tape_linear_linalg_mv, backward_mat_and_vec) {
    /* r = M @ v; loss = sum(r). dM[i,j] = v[j]; dv[j] = sum_i M[i,j]. */
    param_clear();
    double md[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double vd[] = {7.0, 8.0, 9.0};
    int sm[] = {2, 3};
    int sv[] = {3};
    TensorHandle mat = tensor_create(md, sm, 2, 1);
    TensorHandle vec = tensor_create(vd, sv, 1, 1);
    param_register("mat", mat);
    param_register("vec", vec);
    TensorHandle r = tensor_mv(mat, vec);
    TensorHandle loss = tensor_sum(r);
    tensor_backward(loss);
    /* dM[i,j] = v[j] = [7, 8, 9] for each row */
    double expected_M[] = {7, 8, 9, 7, 8, 9};
    for (int i = 0; i < 6; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), expected_M[i], 1e-12,
            "dM[%d] should be %.1f (got %.6f)", i, expected_M[i], param_grad_item_at(0, i));
    /* dv[j] = sum_i M[i,j] = [1+4, 2+5, 3+6] = [5, 7, 9] */
    cr_assert_float_eq(param_grad_item_at(1, 0), 5.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(1, 1), 7.0, 1e-12);
    cr_assert_float_eq(param_grad_item_at(1, 2), 9.0, 1e-12);
}

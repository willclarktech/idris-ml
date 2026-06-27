/* Criterion suite for tape `tensor_view_1d` / `tensor_view_2d` — non-grad
   scalar views into a parent's storage (read-only FFI readback handles). */

#include <criterion/criterion.h>
#include "backend.h"
#include "port_assert.h"

Test(linear_shape_view, view_1d_reads_element) {
	double d[] = {10.0, 20.0, 30.0, 40.0};
	int s[] = {4};
	TensorHandle v = tensor_create(d, s, 1, 0);
	cr_assert_float_eq(tensor_item(tensor_view_1d(v, 0)), 10.0, 1e-12);
	cr_assert_float_eq(tensor_item(tensor_view_1d(v, 2)), 30.0, 1e-12);
	cr_assert_float_eq(tensor_item(tensor_view_1d(v, 3)), 40.0, 1e-12);
}

Test(linear_shape_view, view_2d_reads_element) {
	/* Row-major [2,3] = [[1,2,3],[4,5,6]]; view [row, col]. */
	double d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int s[] = {2, 3};
	TensorHandle m = tensor_create(d, s, 2, 0);
	cr_assert_float_eq(tensor_item(tensor_view_2d(m, 0, 0)), 1.0, 1e-12);
	cr_assert_float_eq(tensor_item(tensor_view_2d(m, 0, 2)), 3.0, 1e-12);
	cr_assert_float_eq(tensor_item(tensor_view_2d(m, 1, 0)), 4.0, 1e-12);
	cr_assert_float_eq(tensor_item(tensor_view_2d(m, 1, 2)), 6.0, 1e-12);
}

/* Regression for the arena-aliasing bug fixed in `6578b81` — exercises the
   double-`tensor_select` chain against a fresh param both before and after
   the optimizer step. The post-step branch (line 1510) is the one that
   used to segfault when arena_reset rewound to reissue the parent struct
   or its data buffer; tensor_create(requires_grad=1) now heap-allocates
   those, so the alias can't fire. */
Test(linear_shape_view, tensor_view) {
    param_clear();
    double wdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int wshape[] = {2, 3};
    TensorHandle wmat = tensor_create(wdata, wshape, 2, 1);
    param_register("wmat", wmat);

    /* Select element [0,1] as a scalar view */
    TensorHandle row0 = tensor_select(wmat, 0, 0);
    TensorHandle elem01 = tensor_select(row0, 0, 1);
    ASSERT_NEAR("view elem[0,1]", tensor_item(elem01), 2.0, 1e-10);

    /* Modify parent via optimizer, check update */
    OptimizerHandle sgd = optimizer_create_sgd(1.0); /* lr=1.0 for easy math */
    /* loss = sum(wmat) so grad = ones */
    TensorHandle wsum = tensor_sum(wmat);
    optimizer_zero_grad(sgd);
    tensor_backward(wsum);
    optimizer_step(sgd);
    /* After step: wmat[0,1] should be 2.0 - 1.0*1.0 = 1.0. Re-creating the
       chain via fresh tensor_selects regression-tests the arena_alloc /
       parent-aliasing fix in tape's select.c (a post-optimizer-step arena
       reset can rewind to wmat's own struct address; the snapshot in
       tensor_select prevents the aliasing memset from corrupting it). */
    ASSERT_NEAR("parent updated", tensor_item(tensor_select(tensor_select(wmat, 0, 0), 0, 1)), 1.0, 1e-10);

    optimizer_free(sgd);
    tensor_free(wmat); tensor_free(wsum);
    param_clear();
}

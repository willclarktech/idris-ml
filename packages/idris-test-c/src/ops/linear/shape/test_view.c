/* Criterion suite for tape `tensor_view_1d` / `tensor_view_2d` — non-grad
   scalar views into a parent's storage (read-only FFI readback handles). */

#include <criterion/criterion.h>
#include "backend.h"

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

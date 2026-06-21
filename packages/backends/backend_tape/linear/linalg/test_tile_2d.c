/* Criterion suite for tensor_tile_2d (forward + backward).
 *
 *   output[i, j] = input[i mod m, j mod n]   over [m*rep0, n*rep1]
 *   d output[i, j] / d input[i mod m, j mod n] = 1, else 0
 *
 *   sum-reduction loss: d loss / d input[i, j] = rep0 * rep1
 *   (each source cell is replicated rep0 * rep1 times in the output).
 *
 * Closes the W3/W4 OP_TILE_2D coverage gap on tape + mlx.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

Test(linear_linalg_tile_2d, forward_tile) {
	/* t = [[1, 2]] tile(2, 3) -> [2, 6]:
	 *   [[1, 2, 1, 2, 1, 2],
	 *    [1, 2, 1, 2, 1, 2]]
	 */
	param_clear();
	double td[] = {1.0, 2.0};
	TensorHandle t = tensor_create_param_2d_f64(1, 2, hcopy(td, 2));
	param_register("t", t);
	TensorHandle r = tensor_tile_2d(t, 2, 3);
	/* Check a few representative cells. */
	cr_assert_float_eq(tensor_item_2d(r, 0, 0), 1.0, TEST_TOL_RELAXED,
	                   "tile[0,0] should be 1 (got %.9f)", tensor_item_2d(r, 0, 0));
	cr_assert_float_eq(tensor_item_2d(r, 0, 1), 2.0, TEST_TOL_RELAXED,
	                   "tile[0,1] should be 2 (got %.9f)", tensor_item_2d(r, 0, 1));
	cr_assert_float_eq(tensor_item_2d(r, 1, 4), 1.0, TEST_TOL_RELAXED,
	                   "tile[1,4] should be 1 (col 4 mod 2 = 0) (got %.9f)",
	                   tensor_item_2d(r, 1, 4));
	cr_assert_float_eq(tensor_item_2d(r, 1, 5), 2.0, TEST_TOL_RELAXED,
	                   "tile[1,5] should be 2 (col 5 mod 2 = 1) (got %.9f)",
	                   tensor_item_2d(r, 1, 5));
}

Test(linear_linalg_tile_2d, forward_f32) {
	/* F32 path: same values as `forward_tile`, exercises the DT_F32 branch
	 * (tile_2d.c lines 31-36, 38-39). Small integers fit exactly in f32;
	 * tensor_item_2d readback carries ~1e-6 error, so assert at an explicit
	 * 1e-5 tolerance (NOT TEST_TOL_RELAXED's tighter F64 bound). */
	param_clear();
	double td[] = {1.0, 2.0};
	TensorHandle t = tensor_create_param_2d_streamed(1, 2, hcopy(td, 2), 0, 14);
	param_register("t", t);
	TensorHandle r = tensor_tile_2d(t, 2, 3);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "tile_2d should propagate F32 tag (got %s)",
	                 tensor_dtype_name(r));
	cr_assert_eq(tensor_size(r, 0), 2);
	cr_assert_eq(tensor_size(r, 1), 6);
	cr_assert_float_eq(tensor_item_2d(r, 0, 0), 1.0, 1e-5, "tile F32[0,0] should be 1 (got %.9f)",
	                   tensor_item_2d(r, 0, 0));
	cr_assert_float_eq(tensor_item_2d(r, 0, 1), 2.0, 1e-5, "tile F32[0,1] should be 2 (got %.9f)",
	                   tensor_item_2d(r, 0, 1));
	cr_assert_float_eq(tensor_item_2d(r, 1, 4), 1.0, 1e-5,
	                   "tile F32[1,4] should be 1 (col 4 mod 2 = 0) (got %.9f)",
	                   tensor_item_2d(r, 1, 4));
	cr_assert_float_eq(tensor_item_2d(r, 1, 5), 2.0, 1e-5,
	                   "tile F32[1,5] should be 2 (col 5 mod 2 = 1) (got %.9f)",
	                   tensor_item_2d(r, 1, 5));
}

Test(linear_linalg_tile_2d, backward_grad_accumulates) {
	/* t = [[1, 2]] tile(2, 3) -> [2, 6]. Loss = sum(tile(t)) = 6 * (1+2) = 18.
	 * Each source cell is replicated rep0 * rep1 = 6 times, so
	 *   d loss / d t[0,0] = 6
	 *   d loss / d t[0,1] = 6  */
	param_clear();
	double td[] = {1.0, 2.0};
	TensorHandle t = tensor_create_param_2d_f64(1, 2, hcopy(td, 2));
	param_register("t", t);
	TensorHandle r = tensor_tile_2d(t, 2, 3);
	TensorHandle loss = tensor_sum(r);
	cr_assert_float_eq(tensor_item(loss), 18.0, TEST_TOL_RELAXED, "loss should be 18 (got %.9f)",
	                   tensor_item(loss));
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 6.0, TEST_TOL_RELAXED,
	                   "grad t[0,0] should be 6 (got %.9f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 6.0, TEST_TOL_RELAXED,
	                   "grad t[0,1] should be 6 (got %.9f)", param_grad_item_at(0, 1));
}

Test(linear_linalg_tile_2d, identity_tile) {
	/* tile(1, 1) should be the identity. */
	param_clear();
	double td[] = {7.0, 8.0, 9.0, 10.0};
	TensorHandle t = tensor_create_param_2d_f64(2, 2, hcopy(td, 4));
	param_register("t", t);
	TensorHandle r = tensor_tile_2d(t, 1, 1);
	cr_assert_float_eq(tensor_item_2d(r, 0, 0), 7.0, TEST_TOL_RELAXED);
	cr_assert_float_eq(tensor_item_2d(r, 0, 1), 8.0, TEST_TOL_RELAXED);
	cr_assert_float_eq(tensor_item_2d(r, 1, 0), 9.0, TEST_TOL_RELAXED);
	cr_assert_float_eq(tensor_item_2d(r, 1, 1), 10.0, TEST_TOL_RELAXED);
}

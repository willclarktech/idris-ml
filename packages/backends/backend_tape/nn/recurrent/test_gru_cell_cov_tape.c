/* Coverage companion for tape `tensor_gru_cell` (gru_cell.c).
 *
 * The base suite (test_gru_cell.c) drives only the F64 forward + the
 * all-grad backward with every gate at 0.5 and hh_n = 0. This file
 * closes the remaining arms:
 *
 *   - is_f32 forward block (gru_cell.c:60-69): real F32 storage reached
 *     via tensor_create_streamed(dtag=14) — the bare tensor_create_f32
 *     alias aborts on tape, so the streamed dtag dispatcher is the only
 *     way to materialise an F32 tensor here.
 *   - no-grad free() arm (gru_cell.c:91-95): requires_grad=0 on every
 *     input → r->requires_grad is false → the zG/rG/nG caches are freed
 *     instead of stashed on a tape entry.
 *   - reset-gate backward with hh_n != 0 (gru_cell.c:116-118): the base
 *     suite sets hh_n = 0 so d_r / d_r_raw collapse to 0; here hh_n = 1
 *     so d_r_raw is genuinely non-zero.
 *
 * nn.GRU equations:
 *   z = sigmoid(ih_z + hh_z), r = sigmoid(ih_r + hh_r)
 *   n = tanh(ih_n + r * hh_n), h' = (1 - z) * n + z * prev
 */

#include <criterion/criterion.h>
#include <math.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TAPE

/* F32 forward, no grad: hits the is_f32 arena block AND the no-grad
   free() arm in one shot. ih = hh = 0, prev = 1 → z = r = 0.5, n = 0,
   h' = 0.5 (exactly representable in float). */
Test(gru_cell_cov, f32_forward_nograd) {
	param_clear();
	int o = 1;
	double ih_data[3] = {0.0, 0.0, 0.0};
	double hh_data[3] = {0.0, 0.0, 0.0};
	double prev_data[1] = {1.0};
	int shape_ih[1] = {3 * o};
	int shape_p[1] = {o};
	/* dtag 14 = real F32 storage on tape; rg = 0. */
	TensorHandle ih = tensor_create_streamed(hcopy(ih_data, 3), shape_ih, 1, 0, 0, 14);
	TensorHandle hh = tensor_create_streamed(hcopy(hh_data, 3), shape_ih, 1, 0, 0, 14);
	TensorHandle prev = tensor_create_streamed(hcopy(prev_data, 1), shape_p, 1, 0, 0, 14);
	cr_assert_str_eq(tensor_dtype_name(ih), "F32");

	TensorHandle h = tensor_gru_cell(ih, hh, prev, o);
	cr_assert_str_eq(tensor_dtype_name(h), "F32");
	cr_assert_float_eq(tensor_item_1d(h, 0), 0.5, 1e-5);
	param_clear();
}

/* F32 forward with a non-trivial n gate: ih_n = 0.5, everything else 0,
   prev = 0 → z = r = 0.5, n = tanh(0.5), h' = 0.5 * tanh(0.5). Confirms
   the is_f32 gate math, not just the degenerate 0.5 case. */
Test(gru_cell_cov, f32_forward_nontrivial) {
	param_clear();
	int o = 1;
	double ih_data[3] = {0.0, 0.0, 0.5};
	double hh_data[3] = {0.0, 0.0, 0.0};
	double prev_data[1] = {0.0};
	int shape_ih[1] = {3 * o};
	int shape_p[1] = {o};
	TensorHandle ih = tensor_create_streamed(hcopy(ih_data, 3), shape_ih, 1, 0, 0, 14);
	TensorHandle hh = tensor_create_streamed(hcopy(hh_data, 3), shape_ih, 1, 0, 0, 14);
	TensorHandle prev = tensor_create_streamed(hcopy(prev_data, 1), shape_p, 1, 0, 0, 14);

	TensorHandle h = tensor_gru_cell(ih, hh, prev, o);
	double expected = 0.5 * tanh(0.5); /* (1 - z) * n + z * prev, z = 0.5, prev = 0 */
	cr_assert_float_eq(tensor_item_1d(h, 0), expected, 1e-5);
	param_clear();
}

/* F64 backward with hh_n = 1 → the reset-gate path (d_r, d_r_raw) is
   non-zero, unlike the base suite. ih = [0,0,0], hh = [0,0,1], prev = 1. */
Test(gru_cell_cov, backward_nonzero_reset) {
	param_clear();
	int o = 1;
	double ih_data[3] = {0.0, 0.0, 0.0};
	double hh_data[3] = {0.0, 0.0, 1.0};
	double prev_data[1] = {1.0};
	int shape_ih[1] = {3 * o};
	int shape_p[1] = {o};
	TensorHandle ih = tensor_create(ih_data, shape_ih, 1, 1);
	TensorHandle hh = tensor_create(hh_data, shape_ih, 1, 1);
	TensorHandle prev = tensor_create(prev_data, shape_p, 1, 1);
	param_register("ih", ih);
	param_register("hh", hh);
	param_register("prev", prev);

	TensorHandle h = tensor_gru_cell(ih, hh, prev, o);
	double n = tanh(0.5); /* tanh(ih_n + r*hh_n) = tanh(0 + 0.5*1) */
	double h_expected = (1.0 - 0.5) * n + 0.5 * 1.0;
	cr_assert_float_eq(tensor_item_1d(h, 0), h_expected, 1e-12);

	TensorHandle loss = tensor_sum(h);
	tensor_backward(loss);

	double z = 0.5, r = 0.5, prev_v = 1.0;
	double d_z_raw = (prev_v - n) * z * (1.0 - z);
	double d_n_pre = (1.0 - z) * (1.0 - n * n);
	double d_ih_n = d_n_pre;
	double d_hh_n = d_n_pre * r;
	double d_r = d_n_pre * 1.0; /* hh_n = 1.0 */
	double d_r_raw = d_r * r * (1.0 - r);
	double d_prev = z;

	/* ih layout [z, r, n] */
	cr_assert_float_eq(param_grad_item_at(0, 0), d_z_raw, 1e-12, "ih_z grad");
	cr_assert_float_eq(param_grad_item_at(0, 1), d_r_raw, 1e-12, "ih_r grad");
	cr_assert_float_eq(param_grad_item_at(0, 2), d_ih_n, 1e-12, "ih_n grad");
	/* hh layout [z, r, n] */
	cr_assert_float_eq(param_grad_item_at(1, 0), d_z_raw, 1e-12, "hh_z grad");
	cr_assert_float_eq(param_grad_item_at(1, 1), d_r_raw, 1e-12, "hh_r grad");
	cr_assert_float_eq(param_grad_item_at(1, 2), d_hh_n, 1e-12, "hh_n grad");
	cr_assert_float_eq(param_grad_item_at(2, 0), d_prev, 1e-12, "prev grad");
	param_clear();
}

#endif /* BACKEND_TAPE */

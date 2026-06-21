/* torch-only Criterion suite for the TensorPair accessors.
 *
 * recurrent/pair_helpers.cpp's tensor_pair_first/second — drive a real pair
 * via tensor_lstm_gates_pair and read both members (value check on the
 * LSTM-gate composition). The pair is owned by all_pairs_torch (freed at the
 * next backend reset); there is no tensor_pair_free (removed — it was a dead,
 * double-free-prone API).
 *
 * torch CPU base dtype is F64; gate values are exact arithmetic over
 * sigmoid/tanh, asserted at a relaxed tolerance. */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

static double* hcopy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

Test(torch_nn_recurrent_pair_helpers, lstm_pair_accessors) {
	const int o = 1;
	/* combined [4*o] = [i, f, g, o] pre-activations, all 0:
	   i_gate = sigmoid(0) = 0.5; f_gate = 0.5; g_gate = tanh(0) = 0;
	   o_gate = 0.5. prev_cell = 2.0.
	   new_cell = f*prev + i*g = 0.5*2 + 0.5*0 = 1.0.
	   new_hidden = o * tanh(new_cell) = 0.5 * tanh(1.0). */
	double cd[] = {0.0, 0.0, 0.0, 0.0};
	TensorHandle combined = tensor_create_1d_f32(4 * o, hcopy(cd, 4), /*rg=*/0);
	double pd[] = {2.0};
	TensorHandle prev_cell = tensor_create_1d_f32(o, hcopy(pd, 1), /*rg=*/0);

	TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
	cr_assert_neq(p, NULL, "tensor_lstm_gates_pair should return a pair");

	TensorHandle hidden = tensor_pair_first(p);
	TensorHandle cell = tensor_pair_second(p);

	double expect_cell = 1.0;
	double expect_hidden = 0.5 * tanh(1.0);
	/* F32 inputs (tensor_create_1d_f32) -> F32 readback; assert at 1e-5. */
	cr_assert_float_eq(tensor_item(cell), expect_cell, 1e-5, "new_cell exp %.6f got %.6f",
	                   expect_cell, tensor_item(cell));
	cr_assert_float_eq(tensor_item(hidden), expect_hidden, 1e-5, "new_hidden exp %.6f got %.6f",
	                   expect_hidden, tensor_item(hidden));
}

#endif /* BACKEND_TORCH */

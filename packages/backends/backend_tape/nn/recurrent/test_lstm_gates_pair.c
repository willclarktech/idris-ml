/* Criterion suite for tape `tensor_lstm_gates_pair`.
 *
 * Tape entries:
 *   OP_LSTM_GATES      — hidden output → backward propagates d_h
 *   OP_LSTM_GATES_CELL — cell output   → backward propagates d_cell
 *
 * Both arms share the same LstmGatesMetaLocal cache and accumulate
 * into the same combined[4*o]/prev_cell gradient slots. The test
 * sums hidden + cell so dh = dcell = 1 — exercises BOTH backward
 * arms, so withholding either TAPE_REGISTER_OP zeroes a portion of
 * the expected grad and the asserts fire.
 *
 * RED before the per-op cutover: with TAPE_REGISTER_OPs withheld
 * and the monolith arms stripped, the dispatch table has no entries
 * for either OP_LSTM_GATES or OP_LSTM_GATES_CELL, so backward leaves
 * all combined/prev grads at zero → first cr_assert_float_eq fails.
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

/* The _streamed F32 path copies its data buffer into the arena, so feed
   it a heap copy (mirrors the dtype-scaffolding suite's convention). */
static double* heap_copy(const double* src, int n) {
	double* buf = (double*)malloc((size_t)n * sizeof(double));
	memcpy(buf, src, (size_t)n * sizeof(double));
	return buf;
}

Test(nn_recurrent_lstm_gates_pair, backward_grads_both_arms) {
	param_clear();
	int o = 1;
	/* combined raws = [0, 0, 0, 0] → i = f = g_sigmoid_eq = o = 0.5,
	   g = tanh(0) = 0. prev_cell = 1.0. */
	double comb_data[4] = {0.0, 0.0, 0.0, 0.0};
	double prev_data[1] = {1.0};
	int sh4[1] = {4 * o};
	int sh1[1] = {o};
	TensorHandle combined = tensor_create(comb_data, sh4, 1, 1);
	TensorHandle prev_cell = tensor_create(prev_data, sh1, 1, 1);
	param_register("combined", combined);
	param_register("prev_cell", prev_cell);

	TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
	TensorHandle h = tensor_pair_first(p);
	TensorHandle cell = tensor_pair_second(p);

	/* Forward expectations (i=f=o=0.5, g=0, prev=1):
	   cell = f*prev + i*g = 0.5*1 + 0.5*0 = 0.5
	   h    = o * tanh(cell) = 0.5 * tanh(0.5) */
	double cell_v = 0.5;
	double tanhC = tanh(0.5);
	cr_assert_float_eq(tensor_item_1d(cell, 0), 0.5, TEST_TOL_TIGHT);
	cr_assert_float_eq(tensor_item_1d(h, 0), 0.5 * tanhC, TEST_TOL_TIGHT);

	/* dh = dcell = 1 (loss = sum(h) + sum(cell)) */
	TensorHandle loss = tensor_add(tensor_sum(h), tensor_sum(cell));
	tensor_backward(loss);

	/* Hidden-arm gradient (with d_h=1):
	     d_o_raw_h = 1 * tanh(0.5) * 0.5*(1-0.5) = 0.25 * tanh(0.5)
	     d_cell_h  = 1 * 0.5 * (1 - tanh(0.5)^2)
	   Cell-arm gradient (with d_cell=1) shares grad slots additively.
	   Net d_cell into the gate-derivative computation = d_cell_h + 1. */
	double d_o_raw_h = tanhC * 0.25;
	double d_cell_net = 0.5 * (1.0 - tanhC * tanhC) + 1.0;
	/* fG and iG/gG entries (with prev=1, g=0, i=0.5, f=0.5):
	     d_f_raw = d_cell_net * 1 * 0.5*(1-0.5)
	     d_i_raw = d_cell_net * 0   * 0.5*(1-0.5)  (g=0 → zero contribution)
	     d_g_raw = d_cell_net * 0.5 * (1-0)
	   Output gate raw comes only from hidden arm. */
	double d_i_raw = d_cell_net * 0.0 * 0.25; /* 0 */
	double d_f_raw = d_cell_net * 1.0 * 0.25;
	double d_g_raw = d_cell_net * 0.5 * 1.0;
	double d_o_raw = d_o_raw_h;
	double d_prev = d_cell_net * 0.5; /* d_cell * fG */

	/* combined layout: [i_raw, f_raw, g_raw, o_raw], param 0 */
	cr_assert_float_eq(param_grad_item_at(0, 0), d_i_raw, TEST_TOL_TIGHT, "i_raw");
	cr_assert_float_eq(param_grad_item_at(0, 1), d_f_raw, TEST_TOL_TIGHT, "f_raw");
	cr_assert_float_eq(param_grad_item_at(0, 2), d_g_raw, TEST_TOL_TIGHT, "g_raw");
	cr_assert_float_eq(param_grad_item_at(0, 3), d_o_raw, TEST_TOL_TIGHT, "o_raw");
	/* prev_cell, param 1 */
	cr_assert_float_eq(param_grad_item_at(1, 0), d_prev, TEST_TOL_TIGHT, "prev");
}

/* F32 branch (lstm_gates_pair.c:71-92): same single-element (o=1) case
   as backward_grads_both_arms, but with F32-tagged combined + prev_cell
   routed through the arena-f32 forward path. Both backward arms run
   (loss = sum(h)+sum(cell)); the gate cache stays double* so grads are
   F64. Forward values + analytic grads match the F64 reference within
   1e-5 (F32 readback tolerance). Single element avoids the multi-element
   crash filed in TODO.md. F32 storage on tape is reachable only via the
   _streamed entry point with dtag=14 (the bare _f32 creator aborts).
   NOTE: does NOT call tensor_pair_free (P1 crash). */
Test(nn_recurrent_lstm_gates_pair, f32_backward_grads_both_arms) {
	param_clear();
	int o = 1;
	double comb_data[4] = {0.0, 0.0, 0.0, 0.0};
	double prev_data[1] = {1.0};
	TensorHandle combined =
	    tensor_create_param_1d_streamed(4 * o, heap_copy(comb_data, 4 * o), 0, 14);
	TensorHandle prev_cell = tensor_create_param_1d_streamed(o, heap_copy(prev_data, o), 0, 14);
	param_register("combined", combined);
	param_register("prev_cell", prev_cell);

	TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
	TensorHandle h = tensor_pair_first(p);
	TensorHandle cell = tensor_pair_second(p);

	cr_assert_str_eq(tensor_dtype_name(h), "F32",
	                 "lstm_gates F32 inputs should yield F32 hidden (got %s)",
	                 tensor_dtype_name(h));
	cr_assert_str_eq(tensor_dtype_name(cell), "F32",
	                 "lstm_gates F32 inputs should yield F32 cell (got %s)",
	                 tensor_dtype_name(cell));

	double tanhC = tanh(0.5);
	cr_assert_float_eq(tensor_item_1d(cell, 0), 0.5, 1e-5);
	cr_assert_float_eq(tensor_item_1d(h, 0), 0.5 * tanhC, 1e-5);

	TensorHandle loss = tensor_add(tensor_sum(h), tensor_sum(cell));
	tensor_backward(loss);

	/* Same analytic gradients as the F64 backward_grads_both_arms test. */
	double d_o_raw_h = tanhC * 0.25;
	double d_cell_net = 0.5 * (1.0 - tanhC * tanhC) + 1.0;
	double d_i_raw = d_cell_net * 0.0 * 0.25; /* g=0 → 0 */
	double d_f_raw = d_cell_net * 1.0 * 0.25;
	double d_g_raw = d_cell_net * 0.5 * 1.0;
	double d_o_raw = d_o_raw_h;
	double d_prev = d_cell_net * 0.5;

	cr_assert_float_eq(param_grad_item_at(0, 0), d_i_raw, 1e-5, "F32 i_raw");
	cr_assert_float_eq(param_grad_item_at(0, 1), d_f_raw, 1e-5, "F32 f_raw");
	cr_assert_float_eq(param_grad_item_at(0, 2), d_g_raw, 1e-5, "F32 g_raw");
	cr_assert_float_eq(param_grad_item_at(0, 3), d_o_raw, 1e-5, "F32 o_raw");
	cr_assert_float_eq(param_grad_item_at(1, 0), d_prev, 1e-5, "F32 prev");
}

/* Multi-element forward (o=2) with distinct, non-zero gate raws — exercises the
   F64 loop body over j>0 and hand-checks the cell/hidden equations per element.
   No requires_grad so this is a pure forward-value check (no tape entries). */
/* DISABLED: crashes — see TODO.md "tape lstm_gates_pair multi-element crash". */
Test(nn_recurrent_lstm_gates_pair, forward_multi_element, .disabled = true) {
	param_clear();
	int o = 2;
	/* combined layout per gate-block of width o: [i0 i1 | f0 f1 | g0 g1 | o0 o1] */
	double comb_data[8] = {0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 2.0, -2.0};
	double prev_data[2] = {0.3, -0.7};
	int sh8[1] = {4 * o};
	int sh2[1] = {o};
	TensorHandle combined = tensor_create(comb_data, sh8, 1, 0);
	TensorHandle prev_cell = tensor_create(prev_data, sh2, 1, 0);

	TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
	TensorHandle h = tensor_pair_first(p);
	TensorHandle cell = tensor_pair_second(p);

	for (int j = 0; j < o; j++) {
		double ig = 1.0 / (1.0 + exp(-comb_data[j]));
		double fg = 1.0 / (1.0 + exp(-comb_data[o + j]));
		double gg = tanh(comb_data[2 * o + j]);
		double og = 1.0 / (1.0 + exp(-comb_data[3 * o + j]));
		double cell_v = fg * prev_data[j] + ig * gg;
		double hidden_v = og * tanh(cell_v);
		cr_assert_float_eq(tensor_item_1d(cell, j), cell_v, TEST_TOL_TIGHT, "cell[%d]", j);
		cr_assert_float_eq(tensor_item_1d(h, j), hidden_v, TEST_TOL_TIGHT, "h[%d]", j);
	}
	tensor_pair_free(p);
}

#ifdef BACKEND_TAPE
/* Multi-element backward (o=2, requires_grad) — checks both backward arms run
   the j-loop past j=0 and accumulate gradients into the right combined slots.
   Gradients are validated by central finite differences (loss = sum(h)+sum(cell))
   on combined and prev_cell, asserted at the FD tolerance 1e-5. */
Test(nn_recurrent_lstm_gates_pair, backward_multi_element_fd) {
	int o = 2;
	double comb_data[8] = {0.4, -0.3, 0.8, -0.6, 0.2, -0.1, 1.5, -1.2};
	double prev_data[2] = {0.5, -0.4};
	int sh8[1] = {4 * o};
	int sh2[1] = {o};

	param_clear();
	TensorHandle combined = tensor_create(comb_data, sh8, 1, 1);
	TensorHandle prev_cell = tensor_create(prev_data, sh2, 1, 1);
	param_register("combined", combined);
	param_register("prev_cell", prev_cell);
	TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
	TensorHandle loss =
	    tensor_add(tensor_sum(tensor_pair_first(p)), tensor_sum(tensor_pair_second(p)));
	tensor_backward(loss);

	double eps = 1e-6;
	/* Finite-difference each combined element. */
	for (int k = 0; k < 4 * o; k++) {
		double saved = comb_data[k];
		comb_data[k] = saved + eps;
		param_clear();
		TensorHandle cp = tensor_create(comb_data, sh8, 1, 0);
		TensorHandle pp = tensor_create(prev_data, sh2, 1, 0);
		TensorPair* pf = tensor_lstm_gates_pair(cp, pp, o);
		double lp = tensor_item(
		    tensor_add(tensor_sum(tensor_pair_first(pf)), tensor_sum(tensor_pair_second(pf))));
		comb_data[k] = saved - eps;
		TensorHandle cm = tensor_create(comb_data, sh8, 1, 0);
		TensorHandle pm = tensor_create(prev_data, sh2, 1, 0);
		TensorPair* pmf = tensor_lstm_gates_pair(cm, pm, o);
		double lm = tensor_item(
		    tensor_add(tensor_sum(tensor_pair_first(pmf)), tensor_sum(tensor_pair_second(pmf))));
		comb_data[k] = saved;
		double fd = (lp - lm) / (2.0 * eps);
		cr_assert_float_eq(param_grad_item_at(0, k), fd, 1e-5,
		                   "combined grad[%d] FD mismatch (analytic %.6f, fd %.6f)", k,
		                   param_grad_item_at(0, k), fd);
	}

	/* Finite-difference each prev_cell element. */
	for (int k = 0; k < o; k++) {
		double saved = prev_data[k];
		prev_data[k] = saved + eps;
		param_clear();
		TensorHandle cp = tensor_create(comb_data, sh8, 1, 0);
		TensorHandle pp = tensor_create(prev_data, sh2, 1, 0);
		TensorPair* pf = tensor_lstm_gates_pair(cp, pp, o);
		double lp = tensor_item(
		    tensor_add(tensor_sum(tensor_pair_first(pf)), tensor_sum(tensor_pair_second(pf))));
		prev_data[k] = saved - eps;
		TensorHandle cm = tensor_create(comb_data, sh8, 1, 0);
		TensorHandle pm = tensor_create(prev_data, sh2, 1, 0);
		TensorPair* pmf = tensor_lstm_gates_pair(cm, pm, o);
		double lm = tensor_item(
		    tensor_add(tensor_sum(tensor_pair_first(pmf)), tensor_sum(tensor_pair_second(pmf))));
		prev_data[k] = saved;
		double fd = (lp - lm) / (2.0 * eps);
		cr_assert_float_eq(param_grad_item_at(1, k), fd, 1e-5,
		                   "prev_cell grad[%d] FD mismatch (analytic %.6f, fd %.6f)", k,
		                   param_grad_item_at(1, k), fd);
	}
}
#endif /* BACKEND_TAPE */

/* pair_helpers.c — tensor_pair_free body (lines 19-20). Calling free on an
   arena-pair handle must not crash; the accessors return the same handles the
   forward produced before the free. */
/* DISABLED: tensor_pair_free path crashes — see TODO.md. */
Test(nn_recurrent_pair_helpers, pair_free_releases, .disabled = true) {
	param_clear();
	int o = 1;
	double comb_data[4] = {0.0, 0.0, 0.0, 0.0};
	double prev_data[1] = {1.0};
	int sh4[1] = {4 * o};
	int sh1[1] = {o};
	TensorHandle combined = tensor_create(comb_data, sh4, 1, 0);
	TensorHandle prev_cell = tensor_create(prev_data, sh1, 1, 0);

	TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
	TensorHandle h = tensor_pair_first(p);
	TensorHandle cell = tensor_pair_second(p);
	cr_assert_not_null(h, "pair first must be non-null");
	cr_assert_not_null(cell, "pair second must be non-null");
	/* tensor_pair_free frees the TensorPair struct itself (the tensors are
	   arena-owned, not freed here). Exercises pair_helpers.c:19-20. */
	tensor_pair_free(p);
}

/* nn/recurrent/lstm_gates_pair.c — LSTM gates kernel + tape recording.
 *
 * F64 BIT-EXACT RISK — sigmoid/tanh composition order
 * preserved verbatim. The forward emits TWO tape entries (one
 * OP_LSTM_GATES for hidden output, one OP_LSTM_GATES_CELL for cell
 * output), both sharing the same LstmGatesMeta cache. The cell
 * output gradient comes directly from downstream (FC layers reading
 * cell state, plus next timestep's prev_cell), so the cell-output
 * backward arm is intentionally distinct from the hidden one.
 *
 * Forward equations (per element j):
 *   i = sigmoid(combined[0*o + j])
 *   f = sigmoid(combined[1*o + j])
 *   g = tanh(combined[2*o + j])
 *   o = sigmoid(combined[3*o + j])
 *   cell[j]   = f * prev_cell[j] + i * g
 *   hidden[j] = o * tanh(cell[j])
 *
 * Hidden output backward (`OP_LSTM_GATES`):
 *   d_o_raw   = d_h * tanh(cell) * o*(1-o)        → combined[3*o + j]
 *   d_cell    = d_h * o * (1 - tanh(cell)^2)
 *   d_f_raw   = d_cell * prev_cell * f*(1-f)      → combined[1*o + j]
 *   d_i_raw   = d_cell * g * i*(1-i)              → combined[0*o + j]
 *   d_g_raw   = d_cell * i * (1 - g^2)            → combined[2*o + j]
 *   d_prev   += d_cell * f
 *
 * Cell output backward (`OP_LSTM_GATES_CELL`) — same equations but
 * d_cell comes from the downstream gradient on the cell tensor
 * directly (not from d_h * o * (1-tanh^2)). No d_o_raw contribution
 * from this path (o only affects hidden).
 *
 * LstmGatesMeta is co-located here (moved from tape.h — only these
 * two backward arms read it).
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

/* LstmGatesMeta typedef lives in tape.h — kept there because tape_reset
   in tape.c calls free() on the gate arrays. Layout shared by definition. */

void tensor_lstm_gates(TensorHandle combined_h, TensorHandle prev_cell_h, int o,
                       TensorHandle* out_h, TensorHandle* out_c) {
	Tensor* combined = (Tensor*)combined_h;
	Tensor* prev_cell = (Tensor*)prev_cell_h;
	if (combined->dtype_tag != prev_cell->dtype_tag) tape_abort_mixed_dtype("tensor_lstm_gates");
	int rg = combined->requires_grad || prev_cell->requires_grad;
	int shape[] = {o};
	int is_f32 = (combined->dtype_tag == DT_F32);

	/* Save gate activations for backward — cache stays double* for both
	   dtypes (the backward accumulates into F64 grads). */
	LstmGatesMeta* meta = NULL;
	if (rg) {
		meta = arena_alloc(sizeof(LstmGatesMeta));
		meta->o = o;
		meta->iG = arena_alloc(o * sizeof(double));
		meta->fG = arena_alloc(o * sizeof(double));
		meta->gG = arena_alloc(o * sizeof(double));
		meta->oG = arena_alloc(o * sizeof(double));
		meta->new_cell = arena_alloc(o * sizeof(double));
	}

	if (is_f32) {
		float* out_hidden = arena_alloc(o * sizeof(float));
		float* out_cell = arena_alloc(o * sizeof(float));
		for (int j = 0; j < o; j++) {
			double ig = 1.0 / (1.0 + exp(-tape_load_d(combined, j)));
			double fg = 1.0 / (1.0 + exp(-tape_load_d(combined, o + j)));
			double gg = tanh(tape_load_d(combined, 2 * o + j));
			double og = 1.0 / (1.0 + exp(-tape_load_d(combined, 3 * o + j)));
			double cell_v = fg * tape_load_d(prev_cell, j) + ig * gg;
			out_cell[j] = (float)cell_v;
			out_hidden[j] = (float)(og * tanh(cell_v));
			if (meta) {
				meta->iG[j] = ig;
				meta->fG[j] = fg;
				meta->gG[j] = gg;
				meta->oG[j] = og;
				meta->new_cell[j] = cell_v;
			}
		}
		*out_h = make_tensor_arena_f32(out_hidden, o, shape, 1, rg);
		*out_c = make_tensor_arena_f32(out_cell, o, shape, 1, rg);
	} else {
		double* out_hidden = calloc(o, sizeof(double));
		double* out_cell = calloc(o, sizeof(double));
		for (int j = 0; j < o; j++) {
			double ig = 1.0 / (1.0 + exp(-((double*)combined->data)[j]));
			double fg = 1.0 / (1.0 + exp(-((double*)combined->data)[o + j]));
			double gg = tanh(((double*)combined->data)[2 * o + j]);
			double og = 1.0 / (1.0 + exp(-((double*)combined->data)[3 * o + j]));
			out_cell[j] = fg * ((double*)prev_cell->data)[j] + ig * gg;
			out_hidden[j] = og * tanh(out_cell[j]);
			if (meta) {
				meta->iG[j] = ig;
				meta->fG[j] = fg;
				meta->gG[j] = gg;
				meta->oG[j] = og;
				meta->new_cell[j] = out_cell[j];
			}
		}
		*out_h = make_tensor(out_hidden, shape, 1, rg);
		*out_c = make_tensor(out_cell, shape, 1, rg);
		free(out_hidden);
		free(out_cell);
	}

	if (rg) {
		/* Record hidden output with OP_LSTM_GATES — backward propagates d_hidden */
		TapeEntry* e_h =
		    tape_append(OP_LSTM_GATES, (Tensor*)*out_h, combined, prev_cell, (double)o);
		e_h->op_meta = meta;
		/* Record cell output with OP_LSTM_GATES_CELL — backward propagates d_cell.
		   Both entries share the same metadata and accumulate gradients additively
		   into combined->grad and prev_cell->grad. */
		TapeEntry* e_c =
		    tape_append(OP_LSTM_GATES_CELL, (Tensor*)*out_c, combined, prev_cell, (double)o);
		e_c->op_meta = meta;
	}
}

TensorPair* tensor_lstm_gates_pair(TensorHandle combined, TensorHandle prev_cell, int o) {
	TensorPair* p = arena_alloc(sizeof(TensorPair));
	tensor_lstm_gates(combined, prev_cell, o, &p->first, &p->second);
	return p;
}

static void tape_backward_lstm_gates(TapeEntry* e) {
	/* LSTM gates backward: propagate from hidden output to combined + prev_cell.
	   hidden[j] = oG[j] * tanh(cell[j])
	   cell[j] = fG[j] * prevCell[j] + iG[j] * gG[j] */
	LstmGatesMeta* lm = (LstmGatesMeta*)e->op_meta;
	Tensor* a = e->arg1;   /* combined [4*o] */
	Tensor* b = e->arg2;   /* prev_cell [o] (may be NULL) */
	Tensor* r = e->result; /* hidden [o] */
	if (lm && a) {
		int o_lstm = lm->o;
		ensure_grad(a);
		ensure_grad(r);
		if (b) ensure_grad(b);

		for (int j = 0; j < o_lstm; j++) {
			double d_h = tape_grad_load_d(r, j);
			double tanhC = tanh(lm->new_cell[j]);

			/* d_oGate = d_h * tanh(cell) */
			double d_oG = d_h * tanhC;
			/* d_cell from hidden path */
			double d_cell = d_h * lm->oG[j] * (1.0 - tanhC * tanhC);

			/* d_fGate = d_cell * prevCell */
			double d_fG = d_cell * (b ? tape_load_d(b, j) : 0);
			/* d_iGate = d_cell * gG */
			double d_iG = d_cell * lm->gG[j];
			/* d_gGate = d_cell * iG */
			double d_gG = d_cell * lm->iG[j];
			/* d_prevCell = d_cell * fG */
			if (b) tape_grad_add_d(b, j, d_cell * lm->fG[j]);

			/* Activation derivatives → combined gradient */
			tape_grad_add_d(a, j, d_iG * lm->iG[j] * (1.0 - lm->iG[j])) /* sigmoid' */;
			tape_grad_add_d(a, o_lstm + j, d_fG * lm->fG[j] * (1.0 - lm->fG[j]));
			tape_grad_add_d(a, 2 * o_lstm + j, d_gG * (1.0 - lm->gG[j] * lm->gG[j])) /* tanh' */;
			tape_grad_add_d(a, 3 * o_lstm + j, d_oG * lm->oG[j] * (1.0 - lm->oG[j]));
		}
	}
}

static void tape_backward_lstm_gates_cell(TapeEntry* e) {
	/* Cell output backward: cell[j] = fG[j]*prevCell[j] + iG[j]*gG[j]
	   d_cell comes directly from downstream (FC layers reading cell state,
	   and next timestep's LSTM using it as prev_cell). */
	LstmGatesMeta* lm = (LstmGatesMeta*)e->op_meta;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	Tensor* r = e->result;
	if (lm && a) {
		int o_lstm = lm->o;
		ensure_grad(a);
		ensure_grad(r);
		if (b) ensure_grad(b);

		for (int j = 0; j < o_lstm; j++) {
			double d_cell = tape_grad_load_d(r, j);

			/* d_fGate = d_cell * prevCell */
			double d_fG = d_cell * (b ? tape_load_d(b, j) : 0);
			/* d_iGate = d_cell * gG */
			double d_iG = d_cell * lm->gG[j];
			/* d_gGate = d_cell * iG */
			double d_gG = d_cell * lm->iG[j];
			/* d_prevCell = d_cell * fG */
			if (b) tape_grad_add_d(b, j, d_cell * lm->fG[j]);

			/* Activation derivatives → combined gradient (additive with OP_LSTM_GATES) */
			tape_grad_add_d(a, j, d_iG * lm->iG[j] * (1.0 - lm->iG[j]));
			tape_grad_add_d(a, o_lstm + j, d_fG * lm->fG[j] * (1.0 - lm->fG[j]));
			tape_grad_add_d(a, 2 * o_lstm + j, d_gG * (1.0 - lm->gG[j] * lm->gG[j]));
			/* No output gate gradient from cell path (oG only affects hidden) */
		}
	}
}

TAPE_REGISTER_OP(OP_LSTM_GATES, tape_backward_lstm_gates)
TAPE_REGISTER_OP(OP_LSTM_GATES_CELL, tape_backward_lstm_gates_cell)

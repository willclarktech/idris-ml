/* nn/recurrent/gru_cell.c — nn.GRU cell (forward + backward).
 *
 * F64 BIT-EXACT RISK — sigmoid/tanh composition order
 * preserved verbatim. Takes ih = W_ih @ x + b_ih and hh = W_hh @ h + b_hh
 * as separate [3*o] vectors (caller computes the two halves).
 *
 *   z = sigmoid(ih_z + hh_z)
 *   r = sigmoid(ih_r + hh_r)
 *   n = tanh(ih_n + r * hh_n)
 *   h' = (1 - z) * n + z * prev
 *
 * Backward (derived from the equations above):
 *   d_z      = dh' * (prev - n)
 *   d_z_raw  = d_z * z * (1-z)        (flows into ih_z and hh_z)
 *   d_n      = dh' * (1-z)
 *   d_n_pre  = d_n * (1-n*n)          (where n_pre = ih_n + r*hh_n)
 *   d_ih_n   = d_n_pre
 *   d_r      = d_n_pre * hh_n
 *   d_hh_n   = d_n_pre * r
 *   d_r_raw  = d_r * r * (1-r)        (flows into ih_r and hh_r)
 *   d_prev   = dh' * z
 *
 * Pre-2026-05-09 this kernel took a single combined = ih + hh and
 * ignored r (simplified GRU); aligned to the standard nn.GRU equation
 * so the example matches what library users expect.
 *
 * GruCellMeta colocated here (was tape.h) — only this op consumes it.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

/* GruCellMeta typedef lives in tape.h — kept there because tape_reset
   in tape.c calls free() on zG/rG/nG when finalizing OP_GRU_CELL tape
   entries. Layout shared by definition. */

TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh, TensorHandle hprev, int o) {
	Tensor* ih = (Tensor*)hih;
	Tensor* hh = (Tensor*)hhh;
	Tensor* prev = (Tensor*)hprev;
	if (ih->dtype_tag != hh->dtype_tag || ih->dtype_tag != prev->dtype_tag)
		tape_abort_mixed_dtype("tensor_gru_cell");
	int shape[] = {o};
	int rg = ih->requires_grad || hh->requires_grad || prev->requires_grad;
	int is_f32 = (ih->dtype_tag == DT_F32);

	/* Meta caches (zG/rG/nG) stay double* — backward writes F64 grads. */
	double* zG = malloc(o * sizeof(double));
	double* rG = malloc(o * sizeof(double));
	double* nG = malloc(o * sizeof(double));

	Tensor* r;
	if (is_f32) {
		float* out = arena_alloc(o * sizeof(float));
		for (int i = 0; i < o; i++) {
			zG[i] = 1.0 / (1.0 + exp(-(tape_load_d(ih, i) + tape_load_d(hh, i))));
			rG[i] = 1.0 / (1.0 + exp(-(tape_load_d(ih, o + i) + tape_load_d(hh, o + i))));
			nG[i] = tanh(tape_load_d(ih, 2 * o + i) + rG[i] * tape_load_d(hh, 2 * o + i));
			double h = (1.0 - zG[i]) * nG[i] + zG[i] * tape_load_d(prev, i);
			out[i] = (float)h;
		}
		r = make_tensor_arena_f32(out, o, shape, 1, rg);
	} else {
		double* out = calloc(o, sizeof(double));
		for (int i = 0; i < o; i++) {
			zG[i] = 1.0 / (1.0 + exp(-(((double*)ih->data)[i] + ((double*)hh->data)[i])));
			rG[i] = 1.0 / (1.0 + exp(-(((double*)ih->data)[o + i] + ((double*)hh->data)[o + i])));
			nG[i] = tanh(((double*)ih->data)[2 * o + i] + rG[i] * ((double*)hh->data)[2 * o + i]);
			out[i] = (1.0 - zG[i]) * nG[i] + zG[i] * ((double*)prev->data)[i];
		}
		r = make_tensor(out, shape, 1, rg);
		free(out);
	}

	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_GRU_CELL, r, ih, hh, 0);
		GruCellMeta* meta = arena_alloc(sizeof(GruCellMeta));
		meta->o = o;
		meta->zG = zG;
		meta->rG = rG;
		meta->nG = nG;
		meta->prev = prev;
		e->op_meta = meta;
	} else {
		free(zG);
		free(rG);
		free(nG);
	}
	return r;
}

static void tape_backward_gru_cell(TapeEntry* e) {
	GruCellMeta* meta = (GruCellMeta*)e->op_meta;
	int oo = meta->o;
	Tensor* ih = e->arg1;
	Tensor* hh = e->arg2;
	Tensor* prev = meta->prev;
	Tensor* r = e->result;
	ensure_grad(r);
	for (int i = 0; i < oo; i++) {
		double dh = tape_grad_load_d(r, i);
		double zv = meta->zG[i];
		double rv = meta->rG[i];
		double nv = meta->nG[i];
		double hh_n_i = tape_load_d(hh, 2 * oo + i);

		double d_z_raw = dh * (tape_load_d(prev, i) - nv) * zv * (1.0 - zv);
		double d_n_pre = dh * (1.0 - zv) * (1.0 - nv * nv);
		double d_r = d_n_pre * hh_n_i;
		double d_r_raw = d_r * rv * (1.0 - rv);
		double d_hh_n = d_n_pre * rv;

		if (ih && ih->requires_grad) {
			ensure_grad(ih);
			tape_grad_add_d(ih, i, d_z_raw);
			tape_grad_add_d(ih, oo + i, d_r_raw);
			tape_grad_add_d(ih, 2 * oo + i, d_n_pre) /* d_ih_n = d_n_pre */;
		}
		if (hh && hh->requires_grad) {
			ensure_grad(hh);
			tape_grad_add_d(hh, i, d_z_raw);
			tape_grad_add_d(hh, oo + i, d_r_raw);
			tape_grad_add_d(hh, 2 * oo + i, d_hh_n);
		}
		if (prev && prev->requires_grad) {
			ensure_grad(prev);
			tape_grad_add_d(prev, i, dh * zv);
		}
	}
}

TAPE_REGISTER_OP(OP_GRU_CELL, tape_backward_gru_cell)

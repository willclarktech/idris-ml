/* conv/avg_pool1d.c — 1D average pooling (forward + backward).
 *
 * Input [C, L], output [C, oL] with oL = (L-kL)/stride+1.
 * Forward: out[c, ol] = mean(input[c, ol*stride .. ol*stride+kL-1]).
 * Backward: gradient = 1/kL spread across the kL contributing positions.
 *
 * AvgPool1DMeta layout shared with tape.h (tape_reset doesn't free
 * anything for this op — meta has only ints).
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
	Tensor* input = (Tensor*)hinput;
	int C = input->shape[0], L = input->shape[1];
	int oL = (L - kL) / stride + 1;
	double scale = 1.0 / kL;
	int is_f32 = (input->dtype_tag == DT_F32);
	int out_shape[] = {C, oL};
	int numel = C * oL;
	void* out =
	    is_f32 ? (void*)arena_alloc(numel * sizeof(float)) : (void*)calloc(numel, sizeof(double));
	for (int c = 0; c < C; c++)
		for (int ol = 0; ol < oL; ol++) {
			double s = 0;
			for (int kl = 0; kl < kL; kl++)
				s += tape_load_d(input, c * L + ol * stride + kl);
			double v = s * scale;
			if (is_f32)
				((float*)out)[c * oL + ol] = (float)v;
			else
				((double*)out)[c * oL + ol] = v;
		}
	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)out, numel, out_shape, 2, input->requires_grad);
	else {
		r = make_tensor((double*)out, out_shape, 2, input->requires_grad);
		free(out);
	}
	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_AVG_POOL1D, r, input, NULL, 0);
		AvgPool1DMeta* meta = arena_alloc(sizeof(AvgPool1DMeta));
		meta->C = C;
		meta->L = L;
		meta->kL = kL;
		meta->stride = stride;
		meta->oL = oL;
		e->op_meta = meta;
	}
	return r;
}

static void tape_backward_avg_pool1d(TapeEntry* e) {
	AvgPool1DMeta* meta = (AvgPool1DMeta*)e->op_meta;
	Tensor* a = e->arg1;
	Tensor* r = e->result;
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		double scale = 1.0 / meta->kL;
		for (int c = 0; c < meta->C; c++)
			for (int ol = 0; ol < meta->oL; ol++)
				for (int kl = 0; kl < meta->kL; kl++)
					tape_grad_add_d(a, c * meta->L + ol * meta->stride + kl,
					                tape_grad_load_d(r, c * meta->oL + ol) * scale);
	}
}

TAPE_REGISTER_OP(OP_AVG_POOL1D, tape_backward_avg_pool1d)

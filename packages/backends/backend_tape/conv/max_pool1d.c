/* conv/max_pool1d.c — 1D max pooling (forward + backward).
 *
 * Input [C, L], output [C, oL] with oL = (L-kL)/stride+1.
 * Forward: out[c, ol] = max(input[c, ol*stride..ol*stride+kL-1]); cache
 * the winning flat-input index per output for backward.
 * Backward: route d_out scalar to the winning input slot (subgradient).
 *
 * MaxPool1DMeta layout stays in tape.h — tape_reset frees max_indices.
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
	Tensor* input = (Tensor*)hinput;
	int C = input->shape[0], L = input->shape[1];
	int oL = (L - kL) / stride + 1;
	int is_f32 = (input->dtype_tag == DT_F32);
	int numel = C * oL;
	int out_shape[] = {C, oL};
	void* out =
	    is_f32 ? (void*)arena_alloc(numel * sizeof(float)) : (void*)calloc(numel, sizeof(double));
	int* max_idx = malloc(numel * sizeof(int));
	for (int c = 0; c < C; c++)
		for (int ol = 0; ol < oL; ol++) {
			double best = -1e30;
			int best_idx = 0;
			for (int kl = 0; kl < kL; kl++) {
				int flat = c * L + (ol * stride + kl);
				double v = tape_load_d(input, flat);
				if (v > best) {
					best = v;
					best_idx = flat;
				}
			}
			if (is_f32)
				((float*)out)[c * oL + ol] = (float)best;
			else
				((double*)out)[c * oL + ol] = best;
			max_idx[c * oL + ol] = best_idx;
		}
	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)out, numel, out_shape, 2, input->requires_grad);
	else {
		r = make_tensor((double*)out, out_shape, 2, input->requires_grad);
		free(out);
	}
	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_MAX_POOL1D, r, input, NULL, 0);
		MaxPool1DMeta* meta = arena_alloc(sizeof(MaxPool1DMeta));
		meta->C = C;
		meta->L = L;
		meta->kL = kL;
		meta->stride = stride;
		meta->oL = oL;
		meta->max_indices = max_idx;
		e->op_meta = meta;
	} else {
		free(max_idx);
	}
	return r;
}

static void tape_backward_max_pool1d(TapeEntry* e) {
	MaxPool1DMeta* meta = (MaxPool1DMeta*)e->op_meta;
	Tensor* a = e->arg1;
	Tensor* r = e->result;
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		int out_numel = meta->C * meta->oL;
		for (int i = 0; i < out_numel; i++)
			tape_grad_add_d(a, meta->max_indices[i], tape_grad_load_d(r, i));
	}
}

TAPE_REGISTER_OP(OP_MAX_POOL1D, tape_backward_max_pool1d)

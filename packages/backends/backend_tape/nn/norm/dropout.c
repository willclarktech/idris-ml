/* nn/norm/dropout.c — inverted dropout (forward + backward).
 *
 * Forward: in training, zero with probability p and scale
 * survivors by 1/(1-p); eval / p=0 returns input identity. Backward
 * applies the same mask. Mask is heap-allocated (survives tape_reset
 * lifecycle by living on tape entry's op_meta).
 *
 * Per-element LCG RNG threaded through seed: deterministic per seed,
 * fast, no shared-state cross-call dependencies.
 */

#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_dropout(TensorHandle hinput, double p, int training, unsigned int seed) {
	Tensor* input = (Tensor*)hinput;
	int n = input->numel;

	if (!training || p <= 0.0) return hinput;

	double scale = 1.0 / (1.0 - p);
	int is_f32 = (input->dtype_tag == DT_F32);
	void* out =
	    is_f32 ? (void*)arena_alloc(n * sizeof(float)) : (void*)arena_alloc(n * sizeof(double));
	double* mask = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++) {
		seed = seed * 1103515245u + 12345u;
		double rv = (double)((seed >> 16) & 0x7fff) / 32767.0;
		if (rv < p) {
			mask[i] = 0.0;
			if (is_f32)
				((float*)out)[i] = 0.0f;
			else
				((double*)out)[i] = 0.0;
		} else {
			mask[i] = scale;
			double v = tape_load_d(input, i) * scale;
			if (is_f32)
				((float*)out)[i] = (float)v;
			else
				((double*)out)[i] = v;
		}
	}

	Tensor* r = arena_alloc(sizeof(Tensor));
	memset(r, 0, sizeof(Tensor));
	r->data = out;
	r->shape = input->shape;
	r->rank = input->rank;
	r->numel = n;
	r->requires_grad = input->requires_grad;
	r->tape_idx = -1;
	r->persistent = 0;
	r->dtype_tag = input->dtype_tag;

	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_DROPOUT, r, input, NULL, 0);
		DropoutMeta* meta = arena_alloc(sizeof(DropoutMeta));
		meta->mask = mask;
		meta->numel = n;
		e->op_meta = meta;
	} else {
		free(mask);
	}
	return r;
}

static void tape_backward_dropout(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	DropoutMeta* meta = (DropoutMeta*)e->op_meta;
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		for (int j = 0; j < meta->numel; j++)
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) * meta->mask[j]);
	}
}

TAPE_REGISTER_OP(OP_DROPOUT, tape_backward_dropout)

/* nn/attention/embedding.c — embedding (forward + backward).
 *
 * Row gather from weight matrix; backward scatters grad
 * rows back. EmbeddingMeta caches indices (heap-allocated, freed in
 * tape_reset).
 */

#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

static Tensor* embedding_impl(Tensor* weight, Tensor* indices, int n, int embedDim, int* out_shape,
                              int out_rank) {
	int out_numel = n * embedDim;
	int* idx_copy = malloc(n * sizeof(int));
	Tensor* r;
	if (weight->dtype_tag == DT_F32) {
		float* out = arena_alloc(out_numel * sizeof(float));
		for (int i = 0; i < n; i++) {
			int idx = (int)tape_load_d(indices, i);
			idx_copy[i] = idx;
			memcpy(out + i * embedDim, ((float*)weight->data) + idx * embedDim,
			       embedDim * sizeof(float));
		}
		r = make_tensor_arena_f32(out, out_numel, out_shape, out_rank, weight->requires_grad);
	} else {
		double* out = calloc(out_numel, sizeof(double));
		for (int i = 0; i < n; i++) {
			int idx = (int)tape_load_d(indices, i);
			idx_copy[i] = idx;
			memcpy(out + i * embedDim, ((double*)weight->data) + idx * embedDim,
			       embedDim * sizeof(double));
		}
		r = make_tensor(out, out_shape, out_rank, weight->requires_grad);
		free(out);
	}
	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_EMBEDDING, r, weight, NULL, 0);
		EmbeddingMeta* meta = arena_alloc(sizeof(EmbeddingMeta));
		meta->n = n;
		meta->embedDim = embedDim;
		meta->indices = idx_copy;
		e->op_meta = meta;
	} else {
		free(idx_copy);
	}
	return r;
}

TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
	Tensor* weight = (Tensor*)hweight;
	Tensor* indices = (Tensor*)hindices;
	int out_shape[] = {n * embedDim};
	return embedding_impl(weight, indices, n, embedDim, out_shape, 1);
}

/* 2D-returning variant: same gather + grad path as tensor_embedding,
 * but the output tensor carries shape [n, embedDim] rather than the
 * flattened [n * embedDim]. The backward closure (tape_backward_embedding)
 * is shape-agnostic — it walks indices and writes to weight's row-major
 * grad buffer, identical for both forward shapes. */
TensorHandle tensor_embedding_2d(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
	Tensor* weight = (Tensor*)hweight;
	Tensor* indices = (Tensor*)hindices;
	int out_shape[] = {n, embedDim};
	return embedding_impl(weight, indices, n, embedDim, out_shape, 2);
}

static void tape_backward_embedding(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	EmbeddingMeta* meta = (EmbeddingMeta*)e->op_meta;
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		for (int i = 0; i < meta->n; i++) {
			int idx = meta->indices[i];
			for (int j = 0; j < meta->embedDim; j++)
				tape_grad_add_d(a, idx * meta->embedDim + j,
				                tape_grad_load_d(r, i * meta->embedDim + j));
		}
	}
}

TAPE_REGISTER_OP(OP_EMBEDDING, tape_backward_embedding)

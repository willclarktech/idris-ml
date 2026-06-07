/* core/lifecycle/lifecycle_ext.c — shape-fixed creation + slicing helpers.
 *
 * The thin shape-fixed wrappers around
 * tensor_create / tensor_reshape that the Idris FFI uses:
 *   - tensor_create_2d (with caller-owned data free)
 *   - tensor_reshape_1d / tensor_reshape_2d
 *   - tensor_item_2d (host-side scalar lookup)
 *   - tensor_one_hot (categorical → one-hot 1D)
 *   - tensor_batch / tensor_unbatch (per-sample <-> batched 4D)
 *   - tensor_subtract_scalar_inplace (mutating scalar broadcast)
 *   - tensor_mul_elementwise / tensor_sum_all (back-compat aliases)
 */

#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
	int shape[] = {rows, cols};
	TensorHandle t = tensor_create(data, shape, 2, requires_grad);
	free(data);
	return t;
}

TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
	int shape[] = {rows, cols};
	return tensor_reshape(h, shape, 2);
}

TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
	int shape[] = {n};
	return tensor_reshape(h, shape, 1);
}

double tensor_item_2d(TensorHandle h, int row, int col) {
	Tensor* t = (Tensor*)h;
	return tape_load_d(t, row * t->shape[1] + col);
}

/* One-hot encode token indices: tokens[n_tokens] → flat [n_tokens * vocab_size].
   `dtag` is currently a no-op (live callers hit F64); see comment. */
TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
	(void)dtag; /* one-hot's 0/1 image fits losslessly in every dtype; today
	               the live callers (Mnist / Gpt / Transformer) hit the F64
	               path, so the dtag is currently a no-op here — when a
	               non-F64 caller appears, route through tape_round_to_dtype
	               + the dtag-keyed arena allocators added in the parity
	               work. */
	int total = n_tokens * vocab_size;
	double* data = calloc(total, sizeof(double));
	for (int i = 0; i < n_tokens; i++) {
		int tok = tokens[i];
		if (tok >= 0 && tok < vocab_size) data[i * vocab_size + tok] = 1.0;
	}
	int shape[] = {total};
	Tensor* r = make_tensor(data, shape, 1, 0);
	free(data);
	free(tokens);
	return r;
}

/* In-place scalar subtraction. Caller mutates the underlying buffer; no tape
   entry — used for non-grad bookkeeping (e.g. centering in unit tests). */
TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
	Tensor* t = (Tensor*)h;
	if (t->dtype_tag == DT_F32) {
		float vf = (float)val;
		for (int i = 0; i < t->numel; i++)
			((float*)t->data)[i] -= vf;
	} else {
		for (int i = 0; i < t->numel; i++)
			((double*)t->data)[i] -= val;
	}
	return h;
}

/* Back-compat aliases — same semantics as the underlying ops. */
TensorHandle tensor_mul_elementwise(TensorHandle ha, TensorHandle hb) {
	return tensor_mul(ha, hb);
}

TensorHandle tensor_sum_all(TensorHandle h) {
	return tensor_sum(h);
}

/* Stack count tensors with identical shape into a fresh [count, ...]
   tensor. dtype must match across inputs; mixed-dtype aborts. */
TensorHandle tensor_batch(TensorHandle* handles, int count) {
	if (count == 0) {
		int shape[] = {0};
		return make_tensor(NULL, shape, 1, 0);
	}
	Tensor* first = (Tensor*)handles[0];
	int elem_size = first->numel;
	int total = elem_size * count;
	int rank = first->rank + 1;
	int is_f32 = (first->dtype_tag == DT_F32);
	for (int i = 1; i < count; i++)
		if (((Tensor*)handles[i])->dtype_tag != first->dtype_tag)
			tape_abort_mixed_dtype("tensor_batch");
	int* shape = malloc(rank * sizeof(int));
	shape[0] = count;
	for (int i = 0; i < first->rank; i++)
		shape[i + 1] = first->shape[i];
	Tensor* r;
	if (is_f32) {
		float* data = arena_alloc(total * sizeof(float));
		for (int i = 0; i < count; i++)
			memcpy(data + (size_t)i * elem_size, ((Tensor*)handles[i])->data,
			       (size_t)elem_size * sizeof(float));
		r = make_tensor_arena_f32(data, total, shape, rank, 0);
	} else {
		double* data = malloc(total * sizeof(double));
		for (int i = 0; i < count; i++)
			memcpy(data + (size_t)i * elem_size, ((Tensor*)handles[i])->data,
			       (size_t)elem_size * sizeof(double));
		r = make_tensor(data, shape, rank, 0);
		free(data);
	}
	free(shape);
	return r;
}

/* Split [B, ...] tensor into B tensors of shape [...].
   Returns array of B tensor handles (caller must free array). */
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
	Tensor* t = (Tensor*)h;
	int B = t->shape[0];
	*out_count = B;
	int elem_size = t->numel / B;
	int inner_rank = t->rank - 1;
	size_t es = tape_elem_size(t->dtype_tag);
	TensorHandle* handles = (TensorHandle*)malloc(B * sizeof(TensorHandle));
	for (int i = 0; i < B; i++) {
		Tensor* r = arena_alloc(sizeof(Tensor));
		memset(r, 0, sizeof(Tensor));
		r->data = (char*)t->data + (size_t)(i * elem_size) * es; /* byte-correct view */
		r->shape = arena_alloc(inner_rank * sizeof(int));
		for (int j = 0; j < inner_rank; j++)
			r->shape[j] = t->shape[j + 1];
		r->rank = inner_rank;
		r->numel = elem_size;
		r->requires_grad = t->requires_grad;
		r->tape_idx = -1;
		r->persistent = 0;
		r->dtype_tag = t->dtype_tag;
		handles[i] = (TensorHandle)r;
	}
	return handles;
}

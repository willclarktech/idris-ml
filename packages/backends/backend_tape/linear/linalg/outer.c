/* linear/linalg/outer.c — outer product of two vectors (forward + backward).
 *
 * r[i,j] = a[i] * b[j].
 * Backward: d_a[i] = sum_j grad[i,j] * b[j]; d_b[j] = sum_i grad[i,j] * a[i].
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_outer(TensorHandle ha, TensorHandle hb) {
	Tensor* a = (Tensor*)ha;
	Tensor* b = (Tensor*)hb;
	if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_outer");
	int m = a->numel, n = b->numel;
	int shape[] = {m, n};
	int rg = a->requires_grad || b->requires_grad;
	if (a->dtype_tag == DT_F32) {
		float* data = arena_alloc((size_t)m * n * sizeof(float));
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				data[(size_t)i * n + j] = ((float*)a->data)[i] * ((float*)b->data)[j];
		Tensor* r = make_tensor_arena_f32(data, m * n, shape, 2, rg);
		if (r->requires_grad) tape_append(OP_OUTER, r, a, b, 0);
		return r;
	}
	double* data = malloc((size_t)m * n * sizeof(double));
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			data[(size_t)i * n + j] = ((double*)a->data)[i] * ((double*)b->data)[j];
	Tensor* r = make_tensor(data, shape, 2, rg);
	free(data);
	if (r->requires_grad) tape_append(OP_OUTER, r, a, b, 0);
	return r;
}

static void tape_backward_outer(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	int m_out = a->numel, n_out = b->numel;
	if (a->requires_grad) {
		ensure_grad(a);
		ensure_grad(r);
		for (int ii = 0; ii < m_out; ii++) {
			double s = 0;
			for (int jj = 0; jj < n_out; jj++)
				s += tape_grad_load_d(r, ii * n_out + jj) * tape_load_d(b, jj);
			tape_grad_add_d(a, ii, s);
		}
	}
	if (b->requires_grad) {
		ensure_grad(b);
		ensure_grad(r);
		for (int jj = 0; jj < n_out; jj++) {
			double s = 0;
			for (int ii = 0; ii < m_out; ii++)
				s += tape_grad_load_d(r, ii * n_out + jj) * tape_load_d(a, ii);
			tape_grad_add_d(b, jj, s);
		}
	}
}

TAPE_REGISTER_OP(OP_OUTER, tape_backward_outer)

/* nn/softmax/softmax.c — softmax (forward + backward).
 *
 * Stable max-subtract formulation. F32 + F64 paths.
 * Backward: d_x_i = sum_j(d_r_j * sm_j * (delta_ij - sm_i)).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

static TensorHandle tensor_softmax_f32(TensorHandle h, int dim) {
	(void)dim;
	Tensor* t = (Tensor*)h;
	int n = t->numel;
	float* td = (float*)t->data;
	float* data = malloc(n * sizeof(float));
	float max_val = td[0];
	for (int i = 1; i < n; i++)
		if (td[i] > max_val) max_val = td[i];
	float sum = 0;
	for (int i = 0; i < n; i++) {
		data[i] = expf(td[i] - max_val);
		sum += data[i];
	}
	for (int i = 0; i < n; i++)
		data[i] /= sum;
	Tensor* r;
	if (t->rank == 0) {
		r = make_scalar_f32((double)data[0], t->requires_grad);
	} else {
		float* arena_d = arena_alloc(n * sizeof(float));
		memcpy(arena_d, data, n * sizeof(float));
		r = make_tensor_arena_f32(arena_d, n, t->shape, t->rank, t->requires_grad);
	}
	free(data);
	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_SOFTMAX, r, t, NULL, 0);
		SoftmaxMeta* meta = arena_alloc(sizeof(SoftmaxMeta));
		meta->n = n;
		meta->out_vals = NULL;
		e->op_meta = meta;
	}
	return r;
}

TensorHandle tensor_softmax(TensorHandle h, int dim) {
	Tensor* t = (Tensor*)h;
	if (t->dtype_tag == DT_F32) return tensor_softmax_f32(h, dim);
	int n = t->numel;
	double* data = malloc(n * sizeof(double));
	double max_val = ((double*)t->data)[0];
	for (int i = 1; i < n; i++)
		if (((double*)t->data)[i] > max_val) max_val = ((double*)t->data)[i];
	double sum = 0;
	for (int i = 0; i < n; i++) {
		data[i] = exp(((double*)t->data)[i] - max_val);
		sum += data[i];
	}
	for (int i = 0; i < n; i++)
		data[i] /= sum;
	Tensor* r;
	if (t->rank == 0) {
		r = make_scalar(data[0], t->requires_grad);
	} else {
		r = make_tensor(data, t->shape, t->rank, t->requires_grad);
	}
	free(data);
	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_SOFTMAX, r, t, NULL, 0);
		SoftmaxMeta* meta = arena_alloc(sizeof(SoftmaxMeta));
		meta->n = n;
		meta->out_vals = r->data;
		e->op_meta = meta;
	}
	return r;
}

static void tape_backward_softmax(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		ensure_grad(r);
		int n_sm = r->numel;
		for (int ii = 0; ii < n_sm; ii++) {
			double sm_i = tape_load_d(r, ii);
			double s = 0;
			for (int jj = 0; jj < n_sm; jj++) {
				double delta = (ii == jj) ? 1.0 : 0.0;
				s += tape_grad_load_d(r, jj) * tape_load_d(r, jj) * (delta - sm_i);
			}
			tape_grad_add_d(a, ii, s);
		}
	}
}

TAPE_REGISTER_OP(OP_SOFTMAX, tape_backward_softmax)

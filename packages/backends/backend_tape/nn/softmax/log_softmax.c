/* nn/softmax/log_softmax.c — 1D log-softmax (forward + backward).
 *
 * Stable max-subtract. F32 + F64 paths.
 * Backward: d_x[j] = grad[j] - exp(r[j]) * sum(grad).
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

static TensorHandle tensor_log_softmax_f32(TensorHandle h, int dim) {
	(void)dim;
	Tensor* t = (Tensor*)h;
	int n = t->numel;
	float* td = (float*)t->data;
	float max_val = td[0];
	for (int i = 1; i < n; i++)
		if (td[i] > max_val) max_val = td[i];
	float sum = 0;
	for (int i = 0; i < n; i++)
		sum += expf(td[i] - max_val);
	float log_sum = logf(sum) + max_val;
	Tensor* r;
	if (t->rank == 0) {
		r = make_scalar_f32((double)(td[0] - log_sum), t->requires_grad);
	} else {
		float* arena_d = arena_alloc(n * sizeof(float));
		for (int i = 0; i < n; i++)
			arena_d[i] = td[i] - log_sum;
		r = make_tensor_arena_f32(arena_d, n, t->shape, t->rank, t->requires_grad);
	}
	if (r->requires_grad) tape_append(OP_LOG_SOFTMAX, r, t, NULL, 0);
	return r;
}

TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
	Tensor* t = (Tensor*)h;
	if (t->dtype_tag == DT_F32) return tensor_log_softmax_f32(h, dim);
	int n = t->numel;
	double* data = malloc(n * sizeof(double));
	double max_val = ((double*)t->data)[0];
	for (int i = 1; i < n; i++)
		if (((double*)t->data)[i] > max_val) max_val = ((double*)t->data)[i];
	double sum = 0;
	for (int i = 0; i < n; i++)
		sum += exp(((double*)t->data)[i] - max_val);
	double log_sum = log(sum) + max_val;
	for (int i = 0; i < n; i++)
		data[i] = ((double*)t->data)[i] - log_sum;
	Tensor* r;
	if (t->rank == 0) {
		r = make_scalar(data[0], t->requires_grad);
	} else {
		r = make_tensor(data, t->shape, t->rank, t->requires_grad);
	}
	free(data);
	if (r->requires_grad) tape_append(OP_LOG_SOFTMAX, r, t, NULL, 0);
	return r;
}

static void tape_backward_log_softmax(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		ensure_grad(r);
		int n_ls = r->numel;
		double sum_grad = 0;
		for (int j = 0; j < n_ls; j++)
			sum_grad += tape_grad_load_d(r, j);
		for (int j = 0; j < n_ls; j++)
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) - exp(tape_load_d(r, j)) * sum_grad);
	}
}

TAPE_REGISTER_OP(OP_LOG_SOFTMAX, tape_backward_log_softmax)

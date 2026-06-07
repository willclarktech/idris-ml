/* nn/softmax/softmax_2d.c — row-wise softmax on 2D (forward + backward). */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_softmax_2d(TensorHandle h) {
	Tensor* t = (Tensor*)h;
	int m = t->shape[0], n = t->shape[1];
	int is_f32 = (t->dtype_tag == DT_F32);
	int shape[] = {m, n};
	void* data =
	    is_f32 ? (void*)arena_alloc(m * n * sizeof(float)) : (void*)malloc(m * n * sizeof(double));
	for (int i = 0; i < m; i++) {
		double max_val = tape_load_d(t, i * n);
		for (int j = 1; j < n; j++) {
			double v = tape_load_d(t, i * n + j);
			if (v > max_val) max_val = v;
		}
		double sum_exp = 0;
		for (int j = 0; j < n; j++) {
			double e = exp(tape_load_d(t, i * n + j) - max_val);
			if (is_f32)
				((float*)data)[i * n + j] = (float)e;
			else
				((double*)data)[i * n + j] = e;
			sum_exp += e;
		}
		for (int j = 0; j < n; j++) {
			if (is_f32)
				((float*)data)[i * n + j] /= (float)sum_exp;
			else
				((double*)data)[i * n + j] /= sum_exp;
		}
	}
	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)data, m * n, shape, 2, t->requires_grad);
	else {
		r = make_tensor((double*)data, shape, 2, t->requires_grad);
		free(data);
	}
	if (r->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, NULL, 0);
	return r;
}

static void tape_backward_softmax_2d(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	int mm = r->shape[0], nn = r->shape[1];
	ensure_grad(r);
	if (a) {
		ensure_grad(a);
		for (int i = 0; i < mm; i++) {
			double dot = 0;
			for (int j = 0; j < nn; j++)
				dot += tape_grad_load_d(r, i * nn + j) * tape_load_d(r, i * nn + j);
			for (int j = 0; j < nn; j++)
				tape_grad_add_d(a, i * nn + j,
				                tape_load_d(r, i * nn + j) *
				                    (tape_grad_load_d(r, i * nn + j) - dot));
		}
	}
}

TAPE_REGISTER_OP(OP_SOFTMAX_2D, tape_backward_softmax_2d)

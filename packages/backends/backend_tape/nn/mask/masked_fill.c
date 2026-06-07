/* nn/mask/masked_fill.c — masked_fill (forward + backward).
 *
 * Forward: r[i] = mask[i] != 0 ? value : t[i].
 * Backward: gradient passes through where mask is 0, zero elsewhere.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle hmask, double value) {
	Tensor* t = (Tensor*)h;
	Tensor* mask = (Tensor*)hmask;
	int n = t->numel;
	Tensor* r;
	if (t->dtype_tag == DT_F32) {
		float* data = arena_alloc(n * sizeof(float));
		float v_f = (float)value;
		for (int i = 0; i < n; i++)
			data[i] = (tape_load_d(mask, i) != 0.0) ? v_f : ((float*)t->data)[i];
		r = make_tensor_arena_f32(data, n, t->shape, t->rank, t->requires_grad);
	} else {
		double* data = malloc(n * sizeof(double));
		for (int i = 0; i < n; i++)
			data[i] = (tape_load_d(mask, i) != 0.0) ? value : ((double*)t->data)[i];
		r = make_tensor(data, t->shape, t->rank, t->requires_grad);
		free(data);
	}
	if (r->requires_grad) tape_append(OP_MASKED_FILL, r, t, mask, 0);
	return r;
}

static void tape_backward_masked_fill(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2; /* mask */
	ensure_grad(r);
	if (a) {
		ensure_grad(a);
		for (int j = 0; j < a->numel; j++)
			if (tape_load_d(b, j) == 0.0) tape_grad_add_d(a, j, tape_grad_load_d(r, j));
	}
}

TAPE_REGISTER_OP(OP_MASKED_FILL, tape_backward_masked_fill)

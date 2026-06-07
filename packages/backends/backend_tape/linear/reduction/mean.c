/* linear/reduction/mean.c — arithmetic mean reducing to a scalar.
 *
 * Forward: scalar = (sum_i t[i]) / numel. Backward: each
 * input element receives grad * (1/numel).
 */

#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_mean(TensorHandle h) {
	Tensor* t = (Tensor*)h;
	double s = 0;
	for (int i = 0; i < t->numel; i++)
		s += tape_load_d(t, i);
	double mean_val = s / t->numel;
	Tensor* r = (t->dtype_tag == DT_F32) ? make_scalar_f32(mean_val, t->requires_grad)
	                                     : make_scalar(mean_val, t->requires_grad);
	if (r->requires_grad) tape_append(OP_MEAN, r, t, NULL, 0);
	return r;
}

static void tape_backward_mean(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		double scale = 1.0 / a->numel;
		for (int j = 0; j < a->numel; j++)
			tape_grad_add_d(a, j, tape_grad_load_d(r, 0) * scale);
	}
}

TAPE_REGISTER_OP(OP_MEAN, tape_backward_mean)

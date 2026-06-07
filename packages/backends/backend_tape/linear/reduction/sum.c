/* linear/reduction/sum.c — element-wise sum reducing to a scalar.
 *
 * Forward: scalar = sum_i t[i]. Backward: grad of every
 * input element += grad of result (the scalar grad broadcasts to all
 * input positions).
 */

#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_sum(TensorHandle h) {
	Tensor* t = (Tensor*)h;
	double s = 0;
	for (int i = 0; i < t->numel; i++)
		s += tape_load_d(t, i);
	Tensor* r = (t->dtype_tag == DT_F32) ? make_scalar_f32(s, t->requires_grad)
	                                     : make_scalar(s, t->requires_grad);
	if (r->requires_grad) tape_append(OP_SUM, r, t, NULL, 0);
	return r;
}

static void tape_backward_sum(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		double rg0 = tape_grad_load_d(r, 0);
		for (int j = 0; j < a->numel; j++)
			tape_grad_add_d(a, j, rg0);
	}
}

TAPE_REGISTER_OP(OP_SUM, tape_backward_sum)

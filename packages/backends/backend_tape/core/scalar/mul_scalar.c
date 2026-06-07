/* core/scalar/mul_scalar.c — element-wise mul-scalar (forward + backward).
 *
 * d(x*s)/dx = s. Scalar arg stored in tape entry's
 * scalar_arg slot and used in backward.
 */

#include <stddef.h> /* NULL */
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

static TensorHandle tensor_mul_scalar_f32(TensorHandle ha, double s) {
	Tensor* a = (Tensor*)ha;
	float sf = (float)s;
	if (a->numel == 1) {
		Tensor* r = make_scalar_f32((double)(((float*)a->data)[0] * sf), a->requires_grad);
		if (r->requires_grad) tape_append(OP_MUL_SCALAR, r, a, NULL, s);
		return r;
	}
	float* data = arena_alloc(a->numel * sizeof(float));
	for (int i = 0; i < a->numel; i++)
		data[i] = ((float*)a->data)[i] * sf;
	Tensor* r = make_tensor_arena_f32(data, a->numel, a->shape, a->rank, a->requires_grad);
	if (r->requires_grad) tape_append(OP_MUL_SCALAR, r, a, NULL, s);
	return r;
}

TensorHandle tensor_mul_scalar(TensorHandle ha, double s) {
	Tensor* a = (Tensor*)ha;
	if (a->dtype_tag == DT_F32) return tensor_mul_scalar_f32(ha, s);
	if (a->numel == 1) {
		Tensor* r = make_scalar(((double*)a->data)[0] * s, a->requires_grad);
		if (r->requires_grad) tape_append(OP_MUL_SCALAR, r, a, NULL, s);
		return r;
	}
	double* data = arena_alloc(a->numel * sizeof(double));
	for (int i = 0; i < a->numel; i++)
		data[i] = ((double*)a->data)[i] * s;
	Tensor* r = make_tensor(data, a->shape, a->rank, a->requires_grad);
	if (r->requires_grad) tape_append(OP_MUL_SCALAR, r, a, NULL, s);
	return r;
}

static void tape_backward_mul_scalar(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		ensure_grad(r);
		for (int j = 0; j < a->numel; j++)
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) * e->scalar_arg);
	}
}

TAPE_REGISTER_OP(OP_MUL_SCALAR, tape_backward_mul_scalar)

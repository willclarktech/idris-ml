/* linear/linalg/transpose_last2.c — transpose last two dims of a 3D tensor.
 *
 * a=[B,m,n] -> r=[B,n,m]. Backward: transpose back.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_transpose_last2(TensorHandle h) {
	Tensor* t = (Tensor*)h;
	int B = t->shape[0], m = t->shape[1], n = t->shape[2];
	int is_f32 = (t->dtype_tag == DT_F32);
	int shape[] = {B, n, m};
	Tensor* r;
	if (is_f32) {
		float* data = arena_alloc(t->numel * sizeof(float));
		for (int bi = 0; bi < B; bi++)
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					data[bi * n * m + j * m + i] = ((float*)t->data)[bi * m * n + i * n + j];
		r = make_tensor_arena_f32(data, t->numel, shape, 3, t->requires_grad);
	} else {
		double* data = malloc(t->numel * sizeof(double));
		for (int bi = 0; bi < B; bi++)
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					data[bi * n * m + j * m + i] = ((double*)t->data)[bi * m * n + i * n + j];
		r = make_tensor(data, shape, 3, t->requires_grad);
		free(data);
	}
	if (t->requires_grad) tape_append(OP_TRANSPOSE_LAST2, r, t, NULL, 0);
	return r;
}

static void tape_backward_transpose_last2(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2];
	ensure_grad(r);
	if (a) {
		ensure_grad(a);
		for (int bi = 0; bi < BB; bi++)
			for (int i = 0; i < mm; i++)
				for (int j = 0; j < nn; j++)
					tape_grad_add_d(a, bi * mm * nn + i * nn + j,
					                tape_grad_load_d(r, bi * nn * mm + j * mm + i));
	}
}

TAPE_REGISTER_OP(OP_TRANSPOSE_LAST2, tape_backward_transpose_last2)

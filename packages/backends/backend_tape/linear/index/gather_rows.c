/* linear/index/gather_rows.c — row-wise gather (forward + backward).
 *
 * Forward: out[i] = input[i*n + index[i]] for i in 0..b-1.
 * Backward: d_input[i*n + index[i]] += d_out[i]; index non-grad
 * (stored as arg2, mirroring OP_GATHER's handling).
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_gather_rows(TensorHandle hinput, TensorHandle hindex, int b, int n) {
	Tensor* input = (Tensor*)hinput;
	Tensor* index = (Tensor*)hindex;
	int shape[] = {b};
	Tensor* r;
	if (input->dtype_tag == DT_F32) {
		float* out = arena_alloc(b * sizeof(float));
		for (int i = 0; i < b; i++) {
			int idx = (int)tape_load_d(index, i);
			out[i] = ((float*)input->data)[(size_t)i * n + idx];
		}
		r = make_tensor_arena_f32(out, b, shape, 1, input->requires_grad);
	} else {
		double* out = calloc(b, sizeof(double));
		for (int i = 0; i < b; i++) {
			int idx = (int)tape_load_d(index, i);
			out[i] = ((double*)input->data)[(size_t)i * n + idx];
		}
		r = make_tensor(out, shape, 1, input->requires_grad);
		free(out);
	}
	if (r->requires_grad) tape_append(OP_GATHER_ROWS, r, input, index, (double)n);
	return r;
}

static void tape_backward_gather_rows(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* index = e->arg2;
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		int n = (int)e->scalar_arg;
		int b = r->numel;
		for (int i = 0; i < b; i++) {
			int idx = (int)tape_load_d(index, i);
			int flat = i * n + idx;
			if (flat >= 0 && flat < a->numel) tape_grad_add_d(a, flat, tape_grad_load_d(r, i));
		}
	}
}

TAPE_REGISTER_OP(OP_GATHER_ROWS, tape_backward_gather_rows)

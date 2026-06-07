/* linear/index/max_rows.c — row-wise max (forward + backward).
 *
 * Forward: out[i] = max_j input[i*n + j] for i in 0..b-1.
 * Backward: d_input[i*n + argmax_i] += d_out[i]; argmax recomputed
 * from the retained arg1 in the backward closure (inputs live on the
 * tape entry — same pattern as OP_GATHER re-reading arg2).
 * Tie-breaking is unspecified across backends (first max wins here).
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_max_rows(TensorHandle hinput, int b, int n) {
	Tensor* input = (Tensor*)hinput;
	int shape[] = {b};
	Tensor* r;
	if (input->dtype_tag == DT_F32) {
		float* out = arena_alloc(b * sizeof(float));
		const float* in = (const float*)input->data;
		for (int i = 0; i < b; i++) {
			float m = in[(size_t)i * n];
			for (int j = 1; j < n; j++)
				if (in[(size_t)i * n + j] > m) m = in[(size_t)i * n + j];
			out[i] = m;
		}
		r = make_tensor_arena_f32(out, b, shape, 1, input->requires_grad);
	} else {
		double* out = calloc(b, sizeof(double));
		const double* in = (const double*)input->data;
		for (int i = 0; i < b; i++) {
			double m = in[(size_t)i * n];
			for (int j = 1; j < n; j++)
				if (in[(size_t)i * n + j] > m) m = in[(size_t)i * n + j];
			out[i] = m;
		}
		r = make_tensor(out, shape, 1, input->requires_grad);
		free(out);
	}
	if (r->requires_grad) tape_append(OP_MAX_ROWS, r, input, NULL, (double)n);
	return r;
}

static void tape_backward_max_rows(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		int n = (int)e->scalar_arg;
		int b = r->numel;
		for (int i = 0; i < b; i++) {
			int am = 0;
			double m = tape_load_d(a, i * n);
			for (int j = 1; j < n; j++) {
				double v = tape_load_d(a, i * n + j);
				if (v > m) {
					m = v;
					am = j;
				}
			}
			tape_grad_add_d(a, i * n + am, tape_grad_load_d(r, i));
		}
	}
}

TAPE_REGISTER_OP(OP_MAX_ROWS, tape_backward_max_rows)

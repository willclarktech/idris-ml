/* core/elementwise/add.c — element-wise addition (forward + backward).
 *
 * First per-op extraction with a real backward. Forward dispatches
 * F32/F64 via binop_elementwise / binop_elementwise_f32_disp (the
 * X-macro stamped kernels live in backend_tape.c for now; the
 * per-TU compile rule moves the .inc to this directory). Backward
 * extracted from the monolith's `case OP_ADD:` arm (was at
 * backend_tape.c lines 3169-3211 pre-extraction) and registered into
 * the global dispatch table via TAPE_REGISTER_OP at file scope — the
 * dispatch-table machinery introduced the table; this is the first
 * op to populate it.
 */

#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../broadcast.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_add(double a, double b) {
	return a + b;
}
static float fn_add_f32(float a, float b) {
	return a + b;
}

TensorHandle tensor_add(TensorHandle a, TensorHandle b) {
	Tensor* ta = (Tensor*)a;
	Tensor* tb = (Tensor*)b;
	if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) {
		if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype("tensor_add");
		return binop_elementwise_f32_disp(a, b, OP_ADD, fn_add_f32);
	}
	return binop_elementwise(a, b, OP_ADD, fn_add);
}

/* Backward: d(a+b)/da = d(a+b)/db = 1. Handles three cases per side:
   same-shape (fast loop), scalar (sum-reduce), and general numpy-style
   broadcast (walk r-positions with broadcast strides, accumulate into
   the operand's flat index). */
static void tape_backward_add(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	int a_match = a && shapes_equal(a, r);
	int b_match = b && shapes_equal(b, r);
	if (a) ensure_grad(a);
	if (b) ensure_grad(b);
	ensure_grad(r);
	if (a_match) {
		for (int j = 0; j < a->numel; j++)
			tape_grad_add_d(a, j, tape_grad_load_d(r, j));
	} else if (a && a->numel == 1) {
		double s = 0;
		for (int j = 0; j < r->numel; j++)
			s += tape_grad_load_d(r, j);
		tape_grad_add_d(a, 0, s);
	}
	if (b_match) {
		for (int j = 0; j < b->numel; j++)
			tape_grad_add_d(b, j, tape_grad_load_d(r, j));
	} else if (b && b->numel == 1) {
		double s = 0;
		for (int j = 0; j < r->numel; j++)
			s += tape_grad_load_d(r, j);
		tape_grad_add_d(b, 0, s);
	}
	if ((a && !a_match && a->numel != 1) || (b && !b_match && b->numel != 1)) {
		int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
		int idx[MAX_BCAST_RANK] = {0};
		if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
		if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
		int do_a = a && !a_match && a->numel != 1;
		int do_b = b && !b_match && b->numel != 1;
		for (int i = 0; i < r->numel; i++) {
			double rg = tape_grad_load_d(r, i);
			if (do_a) {
				int ai = 0;
				for (int k = 0; k < r->rank; k++)
					ai += idx[k] * a_str[k];
				tape_grad_add_d(a, ai, rg);
			}
			if (do_b) {
				int bi = 0;
				for (int k = 0; k < r->rank; k++)
					bi += idx[k] * b_str[k];
				tape_grad_add_d(b, bi, rg);
			}
			/* idx[k] safe: r->rank <= MAX_BCAST_RANK guaranteed by compute_bcast_shape */
			for (int k = r->rank - 1; k >= 0; k--) {
				// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
				if (++idx[k] < r->shape[k]) break;
				idx[k] = 0;
			}
		}
	}
}

/* Constructor-time registration into the global op-dispatch table.
 * After load, tensor_backward's dispatch finds tape_backward_add via
 * tape_dispatch_get(OP_ADD) and calls it without ever entering the
 * monolith switch (which no longer has an OP_ADD case). */
TAPE_REGISTER_OP(OP_ADD, tape_backward_add)

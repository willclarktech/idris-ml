/* core/elementwise/div.c — element-wise division (forward + backward).
 *
 * d(a/b)/da = 1/b, d(a/b)/db = -a/b^2.
 * **F64 BIT-EXACT RISK**: division ordering must be preserved verbatim
 * (load bv once per element, then `/= bv` and `/(bv*bv)`). Don't
 * refactor to fewer divisions — different round-to-nearest sequences
 * would break the byte-identical regression gate.
 */

#include "../../tape.h"
#include "../../arena.h"
#include "../../broadcast.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_div(double a, double b) {
	return a / b;
}
static float fn_div_f32(float a, float b) {
	return a / b;
}

TensorHandle tensor_div(TensorHandle a, TensorHandle b) {
	Tensor* ta = (Tensor*)a;
	Tensor* tb = (Tensor*)b;
	if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) {
		if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype("tensor_div");
		return binop_elementwise_f32_disp(a, b, OP_DIV, fn_div_f32);
	}
	return binop_elementwise(a, b, OP_DIV, fn_div);
}

static void tape_backward_div(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	int a_match = a && shapes_equal(a, r);
	int b_match = b && shapes_equal(b, r);
	if (a) ensure_grad(a);
	if (b) ensure_grad(b);
	ensure_grad(r);
	/* Both forward inputs must be present to compute either backward
	 * gradient (d/da = 1/b needs b; d/db = -a/b² needs both). */
	if (!a || !b) return;
	if (a_match && b_match) {
		for (int j = 0; j < r->numel; j++) {
			double bv = tape_load_d(b, j);
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) / bv);
			tape_grad_add_d(b, j, -(tape_grad_load_d(r, j) * tape_load_d(a, j) / (bv * bv)));
		}
	} else {
		int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
		int idx[MAX_BCAST_RANK] = {0};
		if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
		if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
		for (int i = 0; i < r->numel; i++) {
			int ai = 0, bi = 0;
			for (int k = 0; k < r->rank; k++) {
				ai += idx[k] * a_str[k];
				bi += idx[k] * b_str[k];
			}
			double bv = tape_load_d(b, bi);
			if (a) tape_grad_add_d(a, ai, tape_grad_load_d(r, i) / bv);
			if (b)
				tape_grad_add_d(b, bi, -(tape_grad_load_d(r, i) * tape_load_d(a, ai) / (bv * bv)));
			/* idx[k] safe: r->rank <= MAX_BCAST_RANK guaranteed by compute_bcast_shape */
			for (int k = r->rank - 1; k >= 0; k--) {
				// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
				if (++idx[k] < r->shape[k]) break;
				idx[k] = 0;
			}
		}
	}
}

TAPE_REGISTER_OP(OP_DIV, tape_backward_div)

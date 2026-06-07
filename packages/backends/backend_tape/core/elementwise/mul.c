/* core/elementwise/mul.c — element-wise multiplication (forward + backward).
 *
 * d(a*b)/da = b, d(a*b)/db = a. Inputs are read through
 * tape_load_d so F32 operands work without a separate stamping.
 */

#include "../../tape.h"
#include "../../arena.h"
#include "../../broadcast.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_mul(double a, double b) {
	return a * b;
}
static float fn_mul_f32(float a, float b) {
	return a * b;
}

TensorHandle tensor_mul(TensorHandle a, TensorHandle b) {
	Tensor* ta = (Tensor*)a;
	Tensor* tb = (Tensor*)b;
	if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) {
		if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype("tensor_mul");
		return binop_elementwise_f32_disp(a, b, OP_MUL, fn_mul_f32);
	}
	return binop_elementwise(a, b, OP_MUL, fn_mul);
}

static void tape_backward_mul(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	int a_match = a && shapes_equal(a, r);
	int b_match = b && shapes_equal(b, r);
	if (a) ensure_grad(a);
	if (b) ensure_grad(b);
	ensure_grad(r);
	/* Fast path: both shapes match r */
	if (a_match && b_match) {
		for (int j = 0; j < r->numel; j++) {
			double rg = tape_grad_load_d(r, j);
			if (a && b) tape_grad_add_d(a, j, rg * tape_load_d(b, j));
			if (a && b) tape_grad_add_d(b, j, rg * tape_load_d(a, j));
		}
	} else {
		/* Mixed: scalar / broadcast on either side. Walk r positions. */
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
			double rg = tape_grad_load_d(r, i);
			if (a && b) tape_grad_add_d(a, ai, rg * tape_load_d(b, bi));
			if (a && b) tape_grad_add_d(b, bi, rg * tape_load_d(a, ai));
			/* idx[k] safe: r->rank <= MAX_BCAST_RANK guaranteed by compute_bcast_shape */
			for (int k = r->rank - 1; k >= 0; k--) {
				// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
				if (++idx[k] < r->shape[k]) break;
				idx[k] = 0;
			}
		}
	}
}

TAPE_REGISTER_OP(OP_MUL, tape_backward_mul)

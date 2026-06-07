/* core/elementwise/sub.c — element-wise subtraction (forward + backward).
 *
 * d(a-b)/da = 1, d(a-b)/db = -1 (sign-flipped from add).
 */

#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../broadcast.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_sub(double a, double b) {
	return a - b;
}
static float fn_sub_f32(float a, float b) {
	return a - b;
}

TensorHandle tensor_sub(TensorHandle a, TensorHandle b) {
	Tensor* ta = (Tensor*)a;
	Tensor* tb = (Tensor*)b;
	if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) {
		if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype("tensor_sub");
		return binop_elementwise_f32_disp(a, b, OP_SUB, fn_sub_f32);
	}
	return binop_elementwise(a, b, OP_SUB, fn_sub);
}

static void tape_backward_sub(TapeEntry* e) {
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
			tape_grad_add_d(b, j, -(tape_grad_load_d(r, j)));
	} else if (b && b->numel == 1) {
		double s = 0;
		for (int j = 0; j < r->numel; j++)
			s += tape_grad_load_d(r, j);
		tape_grad_add_d(b, 0, -(s));
	}
	if ((a && !a_match && a->numel != 1) || (b && !b_match && b->numel != 1)) {
		int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
		int idx[MAX_BCAST_RANK] = {0};
		if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
		if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
		int do_a = a && !a_match && a->numel != 1;
		int do_b = b && !b_match && b->numel != 1;
		for (int i = 0; i < r->numel; i++) {
			if (do_a) {
				int ai = 0;
				for (int k = 0; k < r->rank; k++)
					ai += idx[k] * a_str[k];
				tape_grad_add_d(a, ai, tape_grad_load_d(r, i));
			}
			if (do_b) {
				int bi = 0;
				for (int k = 0; k < r->rank; k++)
					bi += idx[k] * b_str[k];
				tape_grad_add_d(b, bi, -(tape_grad_load_d(r, i)));
			}
			for (int k = r->rank - 1; k >= 0; k--) {
				if (++idx[k] < r->shape[k]) break;
				idx[k] = 0;
			}
		}
	}
}

TAPE_REGISTER_OP(OP_SUB, tape_backward_sub)

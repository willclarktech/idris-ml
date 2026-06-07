/* conv/conv1d.c — 1D convolution (forward + backward).
 *
 * Input [inC, L], kernel [outC, inC, kL], bias [outC]
 * or NULL. Output [outC, oL] with oL = (L + 2*pad - kL)/stride + 1.
 *
 *   out[oc, ol] = bias[oc] + sum_{ic,kl} input[ic, ol*stride - pad + kl]
 *                                       * kernel[oc, ic, kl]
 *
 * Bias is passed through TapeEntry's `inputs` slot (cast from Tensor*) —
 * a third gradient channel managed via e->inputs because OP_CONV1D
 * needs three Tensor backrefs, not two. Conv1DMeta layout stays in tape.h.
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias, int pad,
                           int stride) {
	Tensor* input = (Tensor*)hinput;
	Tensor* kernel = (Tensor*)hkernel;
	Tensor* bias = (Tensor*)hbias;
	if (input->dtype_tag != kernel->dtype_tag || (bias && bias->dtype_tag != input->dtype_tag))
		tape_abort_mixed_dtype("tensor_conv1d");
	int inC = input->shape[0], L = input->shape[1];
	int outC = kernel->shape[0], kL = kernel->shape[2];
	int oL = (L + 2 * pad - kL) / stride + 1;
	int is_f32 = (input->dtype_tag == DT_F32);
	int numel = outC * oL;
	int out_shape[] = {outC, oL};
	int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);
	void* out =
	    is_f32 ? (void*)arena_alloc(numel * sizeof(float)) : (void*)calloc(numel, sizeof(double));
	for (int oc = 0; oc < outC; oc++) {
		for (int ol = 0; ol < oL; ol++) {
			double val = bias ? tape_load_d(bias, oc) : 0.0;
			for (int ic = 0; ic < inC; ic++)
				for (int kl = 0; kl < kL; kl++) {
					int il = ol * stride - pad + kl;
					if (il >= 0 && il < L)
						val += tape_load_d(input, ic * L + il) *
						       tape_load_d(kernel, oc * inC * kL + ic * kL + kl);
				}
			if (is_f32)
				((float*)out)[oc * oL + ol] = (float)val;
			else
				((double*)out)[oc * oL + ol] = val;
		}
	}
	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)out, numel, out_shape, 2, rg);
	else {
		r = make_tensor((double*)out, out_shape, 2, rg);
		free(out);
	}
	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_CONV1D, r, input, kernel, 0);
		Conv1DMeta* meta = arena_alloc(sizeof(Conv1DMeta));
		meta->inC = inC;
		meta->outC = outC;
		meta->L = L;
		meta->kL = kL;
		meta->pad = pad;
		meta->stride = stride;
		meta->oL = oL;
		e->op_meta = meta;
		e->inputs = (Tensor**)bias;
	}
	return r;
}

static void tape_backward_conv1d(TapeEntry* e) {
	Conv1DMeta* meta = (Conv1DMeta*)e->op_meta;
	Tensor* a = e->arg1; /* input  */
	Tensor* b = e->arg2; /* kernel */
	Tensor* r = e->result;
	int inC = meta->inC, outC = meta->outC, LL = meta->L;
	int kL = meta->kL, pad = meta->pad, str = meta->stride, oL = meta->oL;
	ensure_grad(r);
	/* d/da needs b->data (kernel), d/db needs a->data (input) — both
	 * inputs required for either gradient. */
	if (a && b && a->requires_grad) {
		ensure_grad(a);
		for (int oc = 0; oc < outC; oc++)
			for (int ol = 0; ol < oL; ol++) {
				double dout = tape_grad_load_d(r, oc * oL + ol);
				for (int ic = 0; ic < inC; ic++)
					for (int kl = 0; kl < kL; kl++) {
						int il = ol * str - pad + kl;
						if (il >= 0 && il < LL)
							tape_grad_add_d(a, ic * LL + il,
							                dout * tape_load_d(b, oc * inC * kL + ic * kL + kl));
					}
			}
	}
	if (a && b && b->requires_grad) {
		ensure_grad(b);
		for (int oc = 0; oc < outC; oc++)
			for (int ic = 0; ic < inC; ic++)
				for (int kl = 0; kl < kL; kl++) {
					double s = 0;
					for (int ol = 0; ol < oL; ol++) {
						int il = ol * str - pad + kl;
						if (il >= 0 && il < LL)
							s += tape_grad_load_d(r, oc * oL + ol) * tape_load_d(a, ic * LL + il);
					}
					tape_grad_add_d(b, oc * inC * kL + ic * kL + kl, s);
				}
	}
	Tensor* bias_t = (Tensor*)e->inputs;
	if (bias_t && bias_t->requires_grad) {
		ensure_grad(bias_t);
		for (int oc = 0; oc < outC; oc++) {
			double s = 0;
			for (int ol = 0; ol < oL; ol++)
				s += tape_grad_load_d(r, oc * oL + ol);
			tape_grad_add_d(bias_t, oc, s);
		}
	}
}

TAPE_REGISTER_OP(OP_CONV1D, tape_backward_conv1d)

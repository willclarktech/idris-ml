/* conv/conv_transpose.c — transposed (fractional-stride) 1D / 2D conv.
 *
 * No backward — these are
 * non-autograd ops on tape (parity with the API surface; mlx/torch
 * keep their native autograd). Used by the encoder/decoder example
 * surfaces that don't drive backward through them.
 *
 *   ConvTranspose1D: out[oc, ol] = bias[oc] + sum_{ic,kl} in[ic, il] * k[ic, oc, kl]
 *                                  where ol = il*stride - pad + kl
 *   ConvTranspose2D: 2D analogue with (kH, kW), (strideH, strideW), (padH, padW).
 *
 * Sum accumulation is done in F64 for stability; F32 outputs narrow at
 * the final store.
 */

#include <stdlib.h>
#include "../arena.h"
#include "../tensor.h"
#include "../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                     int pad, int stride) {
	Tensor* input = (Tensor*)hinput;
	Tensor* kernel = (Tensor*)hkernel;
	Tensor* bias = (Tensor*)hbias;
	if (input->dtype_tag != kernel->dtype_tag || (bias && bias->dtype_tag != input->dtype_tag))
		tape_abort_mixed_dtype("tensor_conv_transpose1d");
	int inC = input->shape[0], L = input->shape[1];
	int outC = kernel->shape[1], kL = kernel->shape[2];
	int oL = (L - 1) * stride - 2 * pad + kL;
	int is_f32 = (input->dtype_tag == DT_F32);
	int numel = outC * oL;
	int out_shape[] = {outC, oL};
	int rg = input->requires_grad || kernel->requires_grad;
	/* Compute in double for sum stability; narrow to float on store. */
	double* dbl = calloc(numel, sizeof(double));
	if (bias)
		for (int oc = 0; oc < outC; oc++)
			for (int ol = 0; ol < oL; ol++)
				dbl[oc * oL + ol] = tape_load_d(bias, oc);
	for (int ic = 0; ic < inC; ic++)
		for (int il = 0; il < L; il++)
			for (int oc = 0; oc < outC; oc++)
				for (int kl = 0; kl < kL; kl++) {
					int ol = il * stride - pad + kl;
					if (ol >= 0 && ol < oL)
						dbl[oc * oL + ol] += tape_load_d(input, ic * L + il) *
						                     tape_load_d(kernel, ic * outC * kL + oc * kL + kl);
				}
	Tensor* r;
	if (is_f32) {
		float* out = arena_alloc(numel * sizeof(float));
		for (int i = 0; i < numel; i++)
			out[i] = (float)dbl[i];
		free(dbl);
		r = make_tensor_arena_f32(out, numel, out_shape, 2, rg);
	} else {
		r = make_tensor(dbl, out_shape, 2, rg);
		free(dbl);
	}
	return r;
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                     int padH, int padW, int strideH, int strideW) {
	Tensor* input = (Tensor*)hinput;
	Tensor* kernel = (Tensor*)hkernel;
	Tensor* bias = (Tensor*)hbias;
	if (input->dtype_tag != kernel->dtype_tag || (bias && bias->dtype_tag != input->dtype_tag))
		tape_abort_mixed_dtype("tensor_conv_transpose2d");
	int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
	int outC = kernel->shape[1], kH = kernel->shape[2], kW = kernel->shape[3];
	int oH = (H - 1) * strideH - 2 * padH + kH;
	int oW = (W - 1) * strideW - 2 * padW + kW;
	int is_f32 = (input->dtype_tag == DT_F32);
	int numel = outC * oH * oW;
	int out_shape[] = {outC, oH, oW};
	int rg = input->requires_grad || kernel->requires_grad;
	double* dbl = calloc(numel, sizeof(double));
	if (bias)
		for (int oc = 0; oc < outC; oc++)
			for (int oh = 0; oh < oH; oh++)
				for (int ow = 0; ow < oW; ow++)
					dbl[oc * oH * oW + oh * oW + ow] = tape_load_d(bias, oc);
	for (int ic = 0; ic < inC; ic++)
		for (int ih = 0; ih < H; ih++)
			for (int iw = 0; iw < W; iw++)
				for (int oc = 0; oc < outC; oc++)
					for (int kh = 0; kh < kH; kh++)
						for (int kw = 0; kw < kW; kw++) {
							int oh = ih * strideH - padH + kh;
							int ow = iw * strideW - padW + kw;
							if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
								dbl[oc * oH * oW + oh * oW + ow] +=
								    tape_load_d(input, ic * H * W + ih * W + iw) *
								    tape_load_d(kernel,
								                ic * outC * kH * kW + oc * kH * kW + kh * kW + kw);
						}
	Tensor* r;
	if (is_f32) {
		float* out = arena_alloc(numel * sizeof(float));
		for (int i = 0; i < numel; i++)
			out[i] = (float)dbl[i];
		free(dbl);
		r = make_tensor_arena_f32(out, numel, out_shape, 3, rg);
	} else {
		r = make_tensor(dbl, out_shape, 3, rg);
		free(dbl);
	}
	return r;
}

/* conv/conv_grouped.c — grouped 1D / 2D convolution (forward only).
 *
 * Forward delegates to ungrouped tensor_conv1d /
 * tensor_conv2d when groups=1. For groups>1, hand-rolled per-group
 * loops — no separate backward; grouped conv on tape doesn't support
 * autograd (mlx/torch use native grouped conv with full autograd).
 *
 * Layout: input [inC, L], kernel [outC, inC_per_group, kL] (or 2D
 * analogue), where inC = inC_per_group * groups and outC must also
 * be divisible by groups.
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                   int pad, int stride, int groups) {
	if (groups == 1) return tensor_conv1d(hinput, hkernel, hbias, pad, stride);
	Tensor* input = (Tensor*)hinput;
	Tensor* kernel = (Tensor*)hkernel;
	Tensor* bias = (Tensor*)hbias;
	if (input->dtype_tag != kernel->dtype_tag || (bias && bias->dtype_tag != input->dtype_tag))
		tape_abort_mixed_dtype("tensor_conv1d_grouped");
	int inC = input->shape[0], L = input->shape[1];
	int outC = kernel->shape[0];
	int inC_g = inC / groups;
	int outC_g = outC / groups;
	int kL = kernel->shape[2];
	int oL = (L + 2 * pad - kL) / stride + 1;
	int total = outC * oL;
	int is_f32 = (input->dtype_tag == DT_F32);
	int out_shape[] = {outC, oL};
	int rg = input->requires_grad || kernel->requires_grad;
	void* out =
	    is_f32 ? (void*)arena_alloc(total * sizeof(float)) : (void*)calloc(total, sizeof(double));
	for (int g = 0; g < groups; g++) {
		for (int oc = 0; oc < outC_g; oc++) {
			int abs_oc = g * outC_g + oc;
			for (int ol = 0; ol < oL; ol++) {
				double val = bias ? tape_load_d(bias, abs_oc) : 0.0;
				for (int ic = 0; ic < inC_g; ic++) {
					int abs_ic = g * inC_g + ic;
					for (int kl = 0; kl < kL; kl++) {
						int il = ol * stride - pad + kl;
						if (il >= 0 && il < L)
							val += tape_load_d(input, abs_ic * L + il) *
							       tape_load_d(kernel, abs_oc * inC_g * kL + ic * kL + kl);
					}
				}
				if (is_f32)
					((float*)out)[abs_oc * oL + ol] = (float)val;
				else
					((double*)out)[abs_oc * oL + ol] = val;
			}
		}
	}
	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)out, total, out_shape, 2, rg);
	else {
		r = make_tensor((double*)out, out_shape, 2, rg);
		free(out);
	}
	return r;
}

TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                   int padH, int padW, int strideH, int strideW, int groups) {
	if (groups == 1) return tensor_conv2d(hinput, hkernel, hbias, padH, padW, strideH, strideW);
	Tensor* input = (Tensor*)hinput;
	Tensor* kernel = (Tensor*)hkernel;
	Tensor* bias = (Tensor*)hbias;
	if (input->dtype_tag != kernel->dtype_tag || (bias && bias->dtype_tag != input->dtype_tag))
		tape_abort_mixed_dtype("tensor_conv2d_grouped");
	int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
	int outC = kernel->shape[0];
	int inC_g = inC / groups;
	int outC_g = outC / groups;
	int kH = kernel->shape[2], kW = kernel->shape[3];
	int oH = (H + 2 * padH - kH) / strideH + 1;
	int oW = (W + 2 * padW - kW) / strideW + 1;
	int numel = outC * oH * oW;
	int is_f32 = (input->dtype_tag == DT_F32);
	int out_shape[] = {outC, oH, oW};
	int rg = input->requires_grad || kernel->requires_grad;
	void* out =
	    is_f32 ? (void*)arena_alloc(numel * sizeof(float)) : (void*)calloc(numel, sizeof(double));
	for (int g = 0; g < groups; g++) {
		for (int oc = 0; oc < outC_g; oc++) {
			int abs_oc = g * outC_g + oc;
			for (int oh = 0; oh < oH; oh++)
				for (int ow = 0; ow < oW; ow++) {
					double val = bias ? tape_load_d(bias, abs_oc) : 0.0;
					for (int ic = 0; ic < inC_g; ic++) {
						int abs_ic = g * inC_g + ic;
						for (int kh = 0; kh < kH; kh++)
							for (int kw = 0; kw < kW; kw++) {
								int ih = oh * strideH - padH + kh;
								int iw = ow * strideW - padW + kw;
								if (ih >= 0 && ih < H && iw >= 0 && iw < W)
									val += tape_load_d(input, abs_ic * H * W + ih * W + iw) *
									       tape_load_d(kernel, abs_oc * inC_g * kH * kW +
									                               ic * kH * kW + kh * kW + kw);
							}
					}
					if (is_f32)
						((float*)out)[abs_oc * oH * oW + oh * oW + ow] = (float)val;
					else
						((double*)out)[abs_oc * oH * oW + oh * oW + ow] = val;
				}
		}
	}
	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)out, numel, out_shape, 3, rg);
	else {
		r = make_tensor((double*)out, out_shape, 3, rg);
		free(out);
	}
	return r;
}

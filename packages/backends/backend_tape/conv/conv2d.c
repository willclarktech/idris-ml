/* conv/conv2d.c — single-sample 2D convolution (forward + backward).
 *
 * Input [inC, H, W], kernel [outC, inC, kH, kW], optional
 * bias [outC]. Output [outC, oH, oW] with the usual valid-window formula
 * (oH = (H + 2*padH - kH)/strideH + 1, similarly for oW).
 *
 *   out[oc, oh, ow] = bias[oc] + sum_{ic,kh,kw} in[ic, ih, iw] * k[oc,ic,kh,kw]
 *
 * Hand-rolled forward + backward (no im2col here — that path is reserved
 * for the batched variant where the dgemm payoff outweighs the unfold
 * cost). Bias is threaded through e->inputs (cast). Conv2DMeta layout
 * stays in tape.h.
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv2d");
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[0], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int out_numel = outC * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);

    void* out = is_f32 ? (void*)arena_alloc(out_numel * sizeof(float))
                       : (void*)calloc(out_numel, sizeof(double));
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double val = bias ? tape_load_d(bias, oc) : 0.0;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * strideH - padH + kh;
                            int iw = ow * strideW - padW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                val += tape_load_d(input, ic*H*W + ih*W + iw)
                                     * tape_load_d(kernel, oc*inC*kH*kW + ic*kH*kW + kh*kW + kw);
                            }
                        }
                    }
                }
                if (is_f32) ((float*)out)[oc*oH*oW + oh*oW + ow] = (float)val;
                else        ((double*)out)[oc*oH*oW + oh*oW + ow] = val;
            }
        }
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, out_numel, out_shape, 3, rg);
    else { r = make_tensor((double*)out, out_shape, 3, rg); free(out); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_CONV2D, r, input, kernel, 0);
        Conv2DMeta* meta = arena_alloc(sizeof(Conv2DMeta));
        meta->inC = inC; meta->outC = outC;
        meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
        /* Store bias pointer in scalar_arg slot (cast) for backward */
        e->inputs = (Tensor**)bias;  /* reuse inputs field for bias ptr */
    }
    return r;
}

static void tape_backward_conv2d(TapeEntry* e) {
    /* r = conv2d(a=input, b=kernel) + bias
       a=[inC,H,W], b=[outC,inC,kH,kW], r=[outC,oH,oW] */
    Conv2DMeta* meta = (Conv2DMeta*)e->op_meta;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    Tensor* r = e->result;
    int inC = meta->inC, outC = meta->outC;
    int HH = meta->H, WW = meta->W, kH = meta->kH, kW = meta->kW;
    int padH = meta->padH, padW = meta->padW;
    int strideH = meta->strH, strideW = meta->strW;
    int oH = meta->oH, oW = meta->oW;
    ensure_grad(r);

    /* d_input — tape_load_d on b->data covers F32 kernels. */
    if (a && a->requires_grad) {
        ensure_grad(a);
        for (int oc = 0; oc < outC; oc++)
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double dout = ((double*)r->grad)[oc*oH*oW + oh*oW + ow];
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < HH && iw >= 0 && iw < WW)
                                    ((double*)a->grad)[ic*HH*WW + ih*WW + iw] +=
                                        dout * tape_load_d(b, oc*inC*kH*kW + ic*kH*kW + kh*kW + kw);
                            }
                }
    }

    /* d_kernel — tape_load_d on a->data covers F32 inputs. */
    if (b && b->requires_grad) {
        ensure_grad(b);
        for (int oc = 0; oc < outC; oc++)
            for (int ic = 0; ic < inC; ic++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        double s = 0;
                        for (int oh = 0; oh < oH; oh++)
                            for (int ow = 0; ow < oW; ow++) {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < HH && iw >= 0 && iw < WW)
                                    s += ((double*)r->grad)[oc*oH*oW + oh*oW + ow]
                                       * tape_load_d(a, ic*HH*WW + ih*WW + iw);
                            }
                        ((double*)b->grad)[oc*inC*kH*kW + ic*kH*kW + kh*kW + kw] += s;
                    }
    }

    /* d_bias */
    Tensor* bias_t = (Tensor*)e->inputs;  /* stored in inputs field */
    if (bias_t && bias_t->requires_grad) {
        ensure_grad(bias_t);
        for (int oc = 0; oc < outC; oc++) {
            double s = 0;
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++)
                    s += ((double*)r->grad)[oc*oH*oW + oh*oW + ow];
            ((double*)bias_t->grad)[oc] += s;
        }
    }
}

TAPE_REGISTER_OP(OP_CONV2D, tape_backward_conv2d)

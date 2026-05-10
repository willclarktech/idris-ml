/* conv/avg_pool2d.c — 2D average pooling (forward + backward).
 *
 * Phase 1d.2.a. Input [C, H, W], output [C, oH, oW] with oH/oW the
 * usual valid-window formula. AvgPool2DMeta stays in tape.h.
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    double scale = 1.0 / (kH * kW);
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {C, oH, oW};
    int numel = C * oH * oW;
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) {
                double s = 0;
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                        s += tape_load_d(input, c*H*W + (oh*strideH+kh)*W + ow*strideW+kw);
                double v = s * scale;
                if (is_f32) ((float*)out)[c*oH*oW + oh*oW + ow] = (float)v;
                else        ((double*)out)[c*oH*oW + oh*oW + ow] = v;
            }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 3, input->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 3, input->requires_grad); free(out); }
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_AVG_POOL2D, r, input, NULL, 0);
        AvgPool2DMeta* meta = arena_alloc(sizeof(AvgPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW; meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
    }
    return r;
}

static void tape_backward_avg_pool2d(TapeEntry* e) {
    AvgPool2DMeta* meta = (AvgPool2DMeta*)e->op_meta;
    Tensor* a = e->arg1;
    Tensor* r = e->result;
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        double scale = 1.0 / (meta->kH * meta->kW);
        for (int c = 0; c < meta->C; c++)
            for (int oh = 0; oh < meta->oH; oh++)
                for (int ow = 0; ow < meta->oW; ow++)
                    for (int kh = 0; kh < meta->kH; kh++)
                        for (int kw = 0; kw < meta->kW; kw++)
                            ((double*)a->grad)[c*meta->H*meta->W + (oh*meta->strH+kh)*meta->W + ow*meta->strW+kw]
                                += ((double*)r->grad)[c*meta->oH*meta->oW + oh*meta->oW + ow] * scale;
    }
}

TAPE_REGISTER_OP(OP_AVG_POOL2D, tape_backward_avg_pool2d)

/* conv/max_pool2d.c — 2D max pooling (forward + backward).
 *
 * Input [C, H, W], output [C, oH, oW]. MaxPool2DMeta
 * carries max_indices (heap-allocated, freed by tape_reset).
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    int out_numel = C * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {C, oH, oW};

    void* out_buf = is_f32 ? (void*)arena_alloc(out_numel * sizeof(float))
                           : (void*)calloc(out_numel, sizeof(double));
    int* max_idx = malloc(out_numel * sizeof(int));

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double best = -1e30;
                int best_idx = 0;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * strideH + kh;
                        int iw = ow * strideW + kw;
                        int flat = c*H*W + ih*W + iw;
                        double v = tape_load_d(input, flat);
                        if (v > best) { best = v; best_idx = flat; }
                    }
                }
                int out_idx = c*oH*oW + oh*oW + ow;
                if (is_f32) ((float*)out_buf)[out_idx] = (float)best;
                else        ((double*)out_buf)[out_idx] = best;
                max_idx[out_idx] = best_idx;
            }
        }
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out_buf, out_numel, out_shape, 3, input->requires_grad);
    else { r = make_tensor((double*)out_buf, out_shape, 3, input->requires_grad); free(out_buf); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MAX_POOL2D, r, input, NULL, 0);
        MaxPool2DMeta* meta = arena_alloc(sizeof(MaxPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;  /* heap-allocated, freed in tape_reset */
        e->op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

static void tape_backward_max_pool2d(TapeEntry* e) {
    MaxPool2DMeta* meta = (MaxPool2DMeta*)e->op_meta;
    Tensor* a = e->arg1;
    Tensor* r = e->result;
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        int out_numel = meta->C * meta->oH * meta->oW;
        for (int i = 0; i < out_numel; i++)
            ((double*)a->grad)[meta->max_indices[i]] += ((double*)r->grad)[i];
    }
}

TAPE_REGISTER_OP(OP_MAX_POOL2D, tape_backward_max_pool2d)

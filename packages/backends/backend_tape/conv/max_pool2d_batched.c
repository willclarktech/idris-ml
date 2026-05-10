/* conv/max_pool2d_batched.c — batched 2D max pooling (forward + backward).
 *
 * Phase 1d.2.c. Input [B, C, H, W], output [B, C, oH, oW]. max_indices
 * are absolute into input->data (computed as base + flat per sample),
 * so the backward scatter works the same as the per-sample case.
 * MaxPool2DBatchedMeta stays in tape.h (max_indices freed by tape_reset).
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW,
                                        int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int B = input->shape[0], C = input->shape[1];
    int H = input->shape[2], W = input->shape[3];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    int out_per_sample = C * oH * oW;
    int out_numel = B * out_per_sample;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {B, C, oH, oW};

    void* out_buf = is_f32 ? (void*)arena_alloc(out_numel * sizeof(float))
                           : (void*)calloc(out_numel, sizeof(double));
    int* max_idx = malloc(out_numel * sizeof(int));

    for (int b = 0; b < B; b++) {
        int base = b * C * H * W;
        int out_base = b * out_per_sample;
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
                            double v = tape_load_d(input, base + flat);
                            if (v > best) { best = v; best_idx = base + flat; }
                        }
                    }
                    int out_idx = c*oH*oW + oh*oW + ow;
                    if (is_f32) ((float*)out_buf)[out_base + out_idx] = (float)best;
                    else        ((double*)out_buf)[out_base + out_idx] = best;
                    max_idx[out_base + out_idx] = best_idx;
                }
            }
        }
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out_buf, out_numel, out_shape, 4, input->requires_grad);
    else { r = make_tensor((double*)out_buf, out_shape, 4, input->requires_grad); free(out_buf); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MAX_POOL2D_BATCHED, r, input, NULL, 0);
        MaxPool2DBatchedMeta* meta = arena_alloc(sizeof(MaxPool2DBatchedMeta));
        meta->B = B; meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;
        e->op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

static void tape_backward_max_pool2d_batched(TapeEntry* e) {
    MaxPool2DBatchedMeta* meta = (MaxPool2DBatchedMeta*)e->op_meta;
    Tensor* a = e->arg1;
    Tensor* r = e->result;
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        int out_numel = meta->B * meta->C * meta->oH * meta->oW;
        for (int i = 0; i < out_numel; i++)
            ((double*)a->grad)[meta->max_indices[i]] += ((double*)r->grad)[i];
    }
}

TAPE_REGISTER_OP(OP_MAX_POOL2D_BATCHED, tape_backward_max_pool2d_batched)

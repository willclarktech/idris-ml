/* nn/softmax/softmax_3d.c — softmax over last dim of [B,m,n]
 * (forward + backward). */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_softmax_3d(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int B = t->shape[0], m = t->shape[1], n = t->shape[2];
    int total_rows = B * m;
    int is_f32 = (t->dtype_tag == DT_F32);
    void* data = is_f32 ? (void*)arena_alloc(t->numel * sizeof(float))
                        : (void*)malloc(t->numel * sizeof(double));
    for (int i = 0; i < total_rows; i++) {
        double max_val = tape_load_d(t, i*n);
        for (int j = 1; j < n; j++) {
            double v = tape_load_d(t, i*n+j);
            if (v > max_val) max_val = v;
        }
        double sum = 0;
        for (int j = 0; j < n; j++) {
            double e = exp(tape_load_d(t, i*n+j) - max_val);
            if (is_f32) ((float*)data)[i*n+j] = (float)e;
            else        ((double*)data)[i*n+j] = e;
            sum += e;
        }
        for (int j = 0; j < n; j++) {
            if (is_f32) ((float*)data)[i*n+j] /= (float)sum;
            else        ((double*)data)[i*n+j] /= sum;
        }
    }
    int shape[] = {B, m, n};
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)data, t->numel, shape, 3, t->requires_grad);
    else { r = make_tensor((double*)data, shape, 3, t->requires_grad); free(data); }
    if (t->requires_grad) tape_append(OP_SOFTMAX_3D, r, t, NULL, 0);
    return r;
}

static void tape_backward_softmax_3d(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2];
    int total_rows = BB * mm;
    ensure_grad(r);
    if (a) {
        ensure_grad(a);
        for (int i = 0; i < total_rows; i++) {
            double dot = 0;
            for (int j = 0; j < nn; j++)
                dot += tape_grad_load_d(r, i*nn+j) * tape_load_d(r, i*nn+j);
            for (int j = 0; j < nn; j++)
                tape_grad_add_d(a, i*nn+j, tape_load_d(r, i*nn+j) * (tape_grad_load_d(r, i*nn+j) - dot));
        }
    }
}

TAPE_REGISTER_OP(OP_SOFTMAX_3D, tape_backward_softmax_3d)

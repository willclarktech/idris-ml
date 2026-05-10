/* linear/index/scatter_add.c — scatter-add by index (forward + backward).
 *
 * Phase 1b.7.b. Forward: r[index[i]] += src[i]. Backward (gather):
 * d_src[i] += d_r[index[i]]. Index stored as arg2.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
    Tensor* index = (Tensor*)hindex;
    Tensor* src = (Tensor*)hsrc;
    int shape[] = {out_size};
    Tensor* r;
    if (src->dtype_tag == DT_F32) {
        float* out = arena_alloc(out_size * sizeof(float));
        for (int i = 0; i < out_size; i++) out[i] = 0.0f;
        for (int i = 0; i < src->numel; i++) {
            int idx = (int)tape_load_d(index, i);
            if (idx >= 0 && idx < out_size)
                out[idx] += ((float*)src->data)[i];
        }
        r = make_tensor_arena_f32(out, out_size, shape, 1, src->requires_grad);
    } else {
        double* out = calloc(out_size, sizeof(double));
        for (int i = 0; i < src->numel; i++) {
            int idx = (int)tape_load_d(index, i);
            if (idx >= 0 && idx < out_size)
                out[idx] += ((double*)src->data)[i];
        }
        r = make_tensor(out, shape, 1, src->requires_grad);
        free(out);
    }
    if (r->requires_grad) tape_append(OP_SCATTER_ADD, r, src, index, (double)out_size);
    return r;
}

static void tape_backward_scatter_add(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;   /* src */
    Tensor* b = e->arg2;   /* index */
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        int nn = a->numel;
        for (int i = 0; i < nn; i++) {
            int idx = (int)tape_load_d(b, i);
            if (idx >= 0 && idx < r->numel)
                ((double*)a->grad)[i] += ((double*)r->grad)[idx];
        }
    }
}

TAPE_REGISTER_OP(OP_SCATTER_ADD, tape_backward_scatter_add)

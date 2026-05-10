/* linear/index/gather.c — gather elements by index (forward + backward).
 *
 * Phase 1b.7. Forward: r[i] = input[index[i]] for i in 0..n-1.
 * Backward (scatter-add): d_input[index[i]] += d_r[i].
 * Index stored as arg2 (non-grad integer tensor).
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    Tensor* input = (Tensor*)hinput;
    Tensor* index = (Tensor*)hindex;
    int shape[] = {n};
    Tensor* r;
    if (input->dtype_tag == DT_F32) {
        float* out = arena_alloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            int idx = (int)tape_load_d(index, i);
            out[i] = ((float*)input->data)[idx];
        }
        r = make_tensor_arena_f32(out, n, shape, 1, input->requires_grad);
    } else {
        double* out = calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) {
            int idx = (int)tape_load_d(index, i);
            out[i] = ((double*)input->data)[idx];
        }
        r = make_tensor(out, shape, 1, input->requires_grad);
        free(out);
    }
    if (r->requires_grad) tape_append(OP_GATHER, r, input, index, (double)n);
    return r;
}

static void tape_backward_gather(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        Tensor* index = b;
        int nn = (int)e->scalar_arg;
        for (int i = 0; i < nn; i++) {
            int idx = (int)tape_load_d(index, i);
            if (idx >= 0 && idx < a->numel)
                ((double*)a->grad)[idx] += ((double*)r->grad)[i];
        }
    }
}

TAPE_REGISTER_OP(OP_GATHER, tape_backward_gather)

/* linear/linalg/transpose_2d.c — 2D transpose (forward + backward).
 *
 * Phase 1b.6. r = a^T where a=[m,n], r=[n,m]. Backward: transpose back.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_transpose_2d(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int m = t->shape[0], n = t->shape[1];
    int shape[] = {n, m};
    Tensor* r;
    if (t->dtype_tag == DT_F32) {
        float* data = arena_alloc(m * n * sizeof(float));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                data[j*m+i] = ((float*)t->data)[i*n+j];
        r = make_tensor_arena_f32(data, m * n, shape, 2, t->requires_grad);
    } else {
        double* data = malloc(m * n * sizeof(double));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                data[j*m+i] = ((double*)t->data)[i*n+j];
        r = make_tensor(data, shape, 2, t->requires_grad);
        free(data);
    }
    if (r->requires_grad) tape_append(OP_TRANSPOSE_2D, r, t, NULL, 0);
    return r;
}

static void tape_backward_transpose_2d(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    int mm = a->shape[0], nn = a->shape[1];
    ensure_grad(r);
    if (a) {
        ensure_grad(a);
        for (int i = 0; i < mm; i++)
            for (int j = 0; j < nn; j++)
                ((double*)a->grad)[i*nn+j] += ((double*)r->grad)[j*mm+i];
    }
}

TAPE_REGISTER_OP(OP_TRANSPOSE_2D, tape_backward_transpose_2d)

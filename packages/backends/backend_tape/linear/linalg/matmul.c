/* linear/linalg/matmul.c — rank-dispatching matrix product.
 *
 * Forward delegates by ranks:
 *   1D x 2D -> OP_VECMAT path here (row vector × matrix)
 *   2D x 1D -> tensor_mv
 *   else    -> elementwise mul fallback (legacy compat)
 *
 * OP_VECMAT backward is co-located here even though it could
 * conceivably live in its own file; the only forward that emits
 * OP_VECMAT is this 1D×2D branch of matmul.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_matmul(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    if (a->rank == 1 && b->rank == 2) {
        if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_matmul");
        int n = a->numel, m = b->shape[1];
        int out_shape[] = {m};
        if (a->dtype_tag == DT_F32) {
            float* out_data = arena_alloc(m * sizeof(float));
            for (int j = 0; j < m; j++) {
                float s = 0;
                for (int i = 0; i < n; i++) s += ((float*)a->data)[i] * ((float*)b->data)[i*m+j];
                out_data[j] = s;
            }
            Tensor* r = make_tensor_arena_f32(out_data, m, out_shape, 1,
                                              a->requires_grad || b->requires_grad);
            if (r->requires_grad) tape_append(OP_VECMAT, r, a, b, 0);
            return r;
        }
        double* out_data = calloc(m, sizeof(double));
        for (int j = 0; j < m; j++) {
            double s = 0;
            for (int i = 0; i < n; i++) s += ((double*)a->data)[i] * ((double*)b->data)[i*m+j];
            out_data[j] = s;
        }
        Tensor* r = make_tensor(out_data, out_shape, 1, a->requires_grad || b->requires_grad);
        free(out_data);
        if (r->requires_grad) tape_append(OP_VECMAT, r, a, b, 0);
        return r;
    }
    if (a->rank == 2 && b->rank == 1) return tensor_mv(ha, hb);
    return tensor_mul(ha, hb);
}

static void tape_backward_vecmat(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    int n_vm = a->numel;
    int m_vm = r->numel;
    ensure_grad(r);
    if (a) {
        ensure_grad(a);
        for (int i = 0; i < n_vm; i++) {
            double s = 0;
            for (int j = 0; j < m_vm; j++) s += ((double*)r->grad)[j] * tape_load_d(b, i*m_vm+j);
            ((double*)a->grad)[i] += s;
        }
    }
    if (b) {
        ensure_grad(b);
        for (int i = 0; i < n_vm; i++) {
            double a_i = tape_load_d(a, i);
            for (int j = 0; j < m_vm; j++)
                ((double*)b->grad)[i*m_vm+j] += ((double*)r->grad)[j] * a_i;
        }
    }
}

TAPE_REGISTER_OP(OP_VECMAT, tape_backward_vecmat)

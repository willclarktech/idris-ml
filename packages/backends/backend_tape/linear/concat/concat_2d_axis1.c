/* linear/concat/concat_2d_axis1.c — 2D concat along axis 1.
 *
 * Phase 1b.2.c. Forward: A[m,n] ++ B[m,k] -> R[m, n+k] interleaved
 * row-by-row. dtype-aware (F32 + F64). scalar_arg stores n (the
 * split point along axis 1).
 *
 * Backward: split R's grad columnwise back to A and B.
 */

#include <string.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_concat_2d_axis1(TensorHandle hA, TensorHandle hB) {
    Tensor* A = (Tensor*)hA;
    Tensor* B = (Tensor*)hB;
    if (A->dtype_tag != B->dtype_tag) tape_abort_mixed_dtype("tensor_concat_2d_axis1");
    int m = A->shape[0];
    int n = A->shape[1];
    int k = B->shape[1];
    int out_shape[] = {m, n + k};
    int rg = A->requires_grad || B->requires_grad;
    if (A->dtype_tag == DT_F32) {
        float* out_data = arena_alloc(m * (n + k) * sizeof(float));
        for (int i = 0; i < m; i++) {
            memcpy(out_data + i * (n + k), ((float*)A->data) + i * n, n * sizeof(float));
            memcpy(out_data + i * (n + k) + n, ((float*)B->data) + i * k, k * sizeof(float));
        }
        Tensor* r = make_tensor_arena_f32(out_data, m * (n + k), out_shape, 2, rg);
        if (rg) tape_append(OP_CONCAT_2D_AXIS1, r, A, B, (double)n);
        return r;
    }
    double* out_data = malloc(m * (n + k) * sizeof(double));
    for (int i = 0; i < m; i++) {
        memcpy(out_data + i * (n + k), ((double*)A->data) + i * n, n * sizeof(double));
        memcpy(out_data + i * (n + k) + n, ((double*)B->data) + i * k, k * sizeof(double));
    }
    Tensor* r = make_tensor(out_data, out_shape, 2, rg);
    free(out_data);
    if (rg) tape_append(OP_CONCAT_2D_AXIS1, r, A, B, (double)n);
    return r;
}

static void tape_backward_concat_2d_axis1(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    int m_c = a->shape[0];
    int n_c = (int)e->scalar_arg;
    int k_c = b->shape[1];
    ensure_grad(r);
    if (a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < m_c; i++)
            for (int j = 0; j < n_c; j++)
                ((double*)a->grad)[i*n_c + j] += ((double*)r->grad)[i*(n_c + k_c) + j];
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int i = 0; i < m_c; i++)
            for (int j = 0; j < k_c; j++)
                ((double*)b->grad)[i*k_c + j] += ((double*)r->grad)[i*(n_c + k_c) + (n_c + j)];
    }
}

TAPE_REGISTER_OP(OP_CONCAT_2D_AXIS1, tape_backward_concat_2d_axis1)

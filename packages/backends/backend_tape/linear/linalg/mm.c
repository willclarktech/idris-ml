/* linear/linalg/mm.c — matrix-matrix multiply (forward + backward).
 *
 * F64 BIT-EXACT RISK — cblas_dgemm arg order, leading
 * dimensions must remain verbatim. Forward: r = a @ b where a=[m,n],
 * b=[n,k], r=[m,k]. Backward: d_a = grad @ b^T; d_b = a^T @ grad.
 * F32 inputs fall back to plain loops via tape_load_d (grad always F64).
 */

#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_mm(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_mm");
    int m = a->shape[0], n = a->shape[1], k = b->shape[1];
    int rg = a->requires_grad || b->requires_grad;
    int shape[] = {m, k};
    if (a->dtype_tag == DT_F32) {
        float* data = arena_alloc(m * k * sizeof(float));
#ifdef __APPLE__
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, n, 1.0f,
                    (const float*)a->data, n, (const float*)b->data, k,
                    0.0f, data, k);
#else
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++) {
                float s = 0;
                for (int p = 0; p < n; p++) s += ((float*)a->data)[i*n+p] * ((float*)b->data)[p*k+j];
                data[i*k+j] = s;
            }
#endif
        Tensor* r = make_tensor_arena_f32(data, m * k, shape, 2, rg);
        if (rg) tape_append(OP_MM, r, a, b, 0);
        return r;
    }
    double* data = calloc(m * k, sizeof(double));
#ifdef __APPLE__
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, k, n, 1.0, a->data, n, b->data, k, 0.0, data, k);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            double s = 0;
            for (int p = 0; p < n; p++) s += ((double*)a->data)[i*n+p] * ((double*)b->data)[p*k+j];
            data[i*k+j] = s;
        }
#endif
    Tensor* r = make_tensor(data, shape, 2, rg);
    free(data);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return r;
}

static void tape_backward_mm(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    int mm = a->shape[0], nn = a->shape[1], kk = r->shape[1];
    int is_f32 = (a->dtype_tag == DT_F32);
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        if (is_f32) {
            for (int i = 0; i < mm; i++)
                for (int j = 0; j < nn; j++) {
                    double s = 0;
                    for (int p = 0; p < kk; p++) s += ((double*)r->grad)[i*kk+p] * tape_load_d(b, j*kk+p);
                    ((double*)a->grad)[i*nn+j] += s;
                }
        } else {
#ifdef __APPLE__
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        mm, nn, kk, 1.0,
                        r->grad, kk, b->data, kk,
                        1.0, a->grad, nn);
#else
            for (int i = 0; i < mm; i++)
                for (int j = 0; j < nn; j++) {
                    double s = 0;
                    for (int p = 0; p < kk; p++) s += ((double*)r->grad)[i*kk+p] * ((double*)b->data)[j*kk+p];
                    ((double*)a->grad)[i*nn+j] += s;
                }
#endif
        }
    }
    if (b && b->requires_grad) {
        ensure_grad(b);
        if (is_f32) {
            for (int j = 0; j < nn; j++)
                for (int p = 0; p < kk; p++) {
                    double s = 0;
                    for (int i = 0; i < mm; i++) s += tape_load_d(a, i*nn+j) * ((double*)r->grad)[i*kk+p];
                    ((double*)b->grad)[j*kk+p] += s;
                }
        } else {
#ifdef __APPLE__
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        nn, kk, mm, 1.0,
                        a->data, nn, r->grad, kk,
                        1.0, b->grad, kk);
#else
            for (int j = 0; j < nn; j++)
                for (int p = 0; p < kk; p++) {
                    double s = 0;
                    for (int i = 0; i < mm; i++) s += ((double*)a->data)[i*nn+j] * ((double*)r->grad)[i*kk+p];
                    ((double*)b->grad)[j*kk+p] += s;
                }
#endif
        }
    }
}

TAPE_REGISTER_OP(OP_MM, tape_backward_mm)

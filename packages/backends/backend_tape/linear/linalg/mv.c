/* linear/linalg/mv.c — matrix-vector multiply (forward + backward).
 *
 * r = mat @ vec where mat=[m,n], vec=[n], r=[m].
 * Uses BLAS sgemv/dgemv on Apple Accelerate. Caches x_vals (vec
 * snapshot) in MvMeta as double* for the backward (which always
 * reads in F64, even when mat is F32-tagged).
 *
 * Backward: d_mat = grad * x^T (rank-1 outer product); d_vec = mat^T @ grad.
 * F32 mat falls back to plain double loops via tape_load_d; F64 uses
 * cblas_dger (rank-1) and cblas_dgemv (transposed gemv).
 */

#include <string.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

static TensorHandle tensor_mv_f32(TensorHandle hmat, TensorHandle hvec) {
    Tensor* mat = (Tensor*)hmat;
    Tensor* vec = (Tensor*)hvec;
    int m = mat->shape[0], n = mat->shape[1];
    int out_shape[] = {m};
    /* Zero-dim guard (see mm.c). cblas_sgemv rejects lda=0. */
    if (m == 0 || n == 0)
        return tape_zero_tensor(out_shape, 1, DT_F32,
                                mat->requires_grad || vec->requires_grad);
    float* out_data = arena_alloc(m * sizeof(float));
#ifdef __APPLE__
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f,
                (const float*)mat->data, n, (const float*)vec->data, 1,
                0.0f, out_data, 1);
#else
    for (int i = 0; i < m; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) s += ((float*)mat->data)[i*n+j] * ((float*)vec->data)[j];
        out_data[i] = s;
    }
#endif
    Tensor* r = make_tensor_arena_f32(out_data, m, out_shape, 1,
                                      mat->requires_grad || vec->requires_grad);
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MV, r, mat, vec, 0);
        MvMeta* meta = arena_alloc(sizeof(MvMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        for (int j = 0; j < n; j++) meta->x_vals[j] = (double)((float*)vec->data)[j];
        e->op_meta = meta;
    }
    return r;
}

TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
    Tensor* mat = (Tensor*)hmat;
    Tensor* vec = (Tensor*)hvec;
    if (mat->dtype_tag == DT_F32 || vec->dtype_tag == DT_F32) {
        if (mat->dtype_tag != vec->dtype_tag) tape_abort_mixed_dtype("tensor_mv");
        return tensor_mv_f32(hmat, hvec);
    }
    int m = mat->shape[0], n = mat->shape[1];
    int out_shape[] = {m};
    /* Zero-dim guard (see mm.c). cblas_dgemv rejects lda=0. */
    if (m == 0 || n == 0)
        return tape_zero_tensor(out_shape, 1, DT_F64,
                                mat->requires_grad || vec->requires_grad);
    double* out_data = arena_alloc(m * sizeof(double));
#ifdef __APPLE__
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
                mat->data, n, vec->data, 1, 0.0, out_data, 1);
#else
    for (int i = 0; i < m; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += ((double*)mat->data)[i*n+j] * ((double*)vec->data)[j];
        out_data[i] = s;
    }
#endif
    Tensor* r = make_tensor_arena(out_data, m, out_shape, 1,
                                  mat->requires_grad || vec->requires_grad);
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MV, r, mat, vec, 0);
        MvMeta* meta = arena_alloc(sizeof(MvMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        memcpy(meta->x_vals, vec->data, n * sizeof(double));
        e->op_meta = meta;
    }
    return r;
}

static void tape_backward_mv(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    MvMeta* meta = (MvMeta*)e->op_meta;
    int m_mv = meta ? meta->m : a->shape[0];
    int n_mv = meta ? meta->n : a->shape[1];
    double* x_vals = meta ? meta->x_vals : b->data;
    ensure_grad(r);
    if (a->requires_grad) {
        ensure_grad(a);
        if (a->dtype_tag == DT_F32) {
            for (int ii = 0; ii < m_mv; ii++)
                for (int jj = 0; jj < n_mv; jj++)
                    ((double*)a->grad)[ii*n_mv+jj] += ((double*)r->grad)[ii] * x_vals[jj];
        } else {
#ifdef __APPLE__
            /* A.grad [m,n] += grad [m] * x^T [n] — rank-1 outer product */
            cblas_dger(CblasRowMajor, m_mv, n_mv, 1.0,
                       r->grad, 1, x_vals, 1,
                       a->grad, n_mv);
#else
            for (int ii = 0; ii < m_mv; ii++)
                for (int jj = 0; jj < n_mv; jj++)
                    ((double*)a->grad)[ii*n_mv+jj] += ((double*)r->grad)[ii] * x_vals[jj];
#endif
        }
    }
    if (b && b->requires_grad) {
        ensure_grad(b);
        if (a->dtype_tag == DT_F32) {
            /* x.grad [n] += A^T [n,m] @ grad [m], read A as F32 */
            for (int jj = 0; jj < n_mv; jj++) {
                double s = 0;
                for (int ii = 0; ii < m_mv; ii++)
                    s += tape_load_d(a, ii*n_mv+jj) * ((double*)r->grad)[ii];
                ((double*)b->grad)[jj] += s;
            }
        } else {
#ifdef __APPLE__
            cblas_dgemv(CblasRowMajor, CblasTrans, m_mv, n_mv, 1.0,
                        a->data, n_mv, r->grad, 1,
                        1.0, b->grad, 1);
#else
            for (int jj = 0; jj < n_mv; jj++) {
                double s = 0;
                for (int ii = 0; ii < m_mv; ii++)
                    s += ((double*)a->data)[ii*n_mv+jj] * ((double*)r->grad)[ii];
                ((double*)b->grad)[jj] += s;
            }
#endif
        }
    }
}

TAPE_REGISTER_OP(OP_MV, tape_backward_mv)

/* linear/linalg/linear.c — fused linear (W @ x + bias) (forward + backward).
 *
 * F64 BIT-EXACT RISK — cblas_dgemv arg order. Caches x_vals
 * in LinearMeta (double*) and a bias Tensor* pointer. Backward:
 *   dW    = grad * x^T  (cblas_dger rank-1; or F32 plain loop)
 *   dx    = W^T @ grad  (cblas_dgemv transposed; or F32 tape_load_d loop)
 *   dbias = grad        (simple per-element accumulate)
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

static TensorHandle tensor_linear_f32(TensorHandle hW, TensorHandle hx, TensorHandle hbias) {
    Tensor* W = (Tensor*)hW;
    Tensor* x = (Tensor*)hx;
    Tensor* bias = (Tensor*)hbias;
    int m = W->shape[0], n = W->shape[1];
    int out_shape[] = {m};
    /* Zero-dim guard (see mm.c). cblas_sgemv rejects lda=0; n=0 reduces to
     * just adding the bias (or zero if no bias). */
    if (m == 0 || n == 0) {
        int rg0 = W->requires_grad || x->requires_grad || (bias && bias->requires_grad);
        Tensor* r = tape_zero_tensor(out_shape, 1, DT_F32, rg0);
        if (bias && m > 0) {
            /* r += bias (n=0 case where the matmul drops out but bias remains) */
            for (int i = 0; i < m; i++)
                ((float*)r->data)[i] = ((float*)bias->data)[i];
        }
        return r;
    }
    float* out_data = arena_alloc(m * sizeof(float));
#ifdef __APPLE__
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f,
                (const float*)W->data, n, (const float*)x->data, 1,
                0.0f, out_data, 1);
#else
    for (int i = 0; i < m; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) s += ((float*)W->data)[i*n+j] * ((float*)x->data)[j];
        out_data[i] = s;
    }
#endif
    if (bias) {
#ifdef __APPLE__
        vDSP_vadd(out_data, 1, (const float*)bias->data, 1, out_data, 1, (vDSP_Length)m);
#else
        for (int i = 0; i < m; i++) out_data[i] += ((float*)bias->data)[i];
#endif
    }
    int rg = W->requires_grad || x->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor_arena_f32(out_data, m, out_shape, 1, rg);
    if (rg) {
        TapeEntry* e = tape_append(OP_LINEAR, r, W, x, 0);
        LinearMeta* meta = arena_alloc(sizeof(LinearMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        for (int j = 0; j < n; j++) meta->x_vals[j] = (double)((float*)x->data)[j];
        meta->bias = bias;
        e->op_meta = meta;
    }
    return r;
}

TensorHandle tensor_linear(TensorHandle hW, TensorHandle hx, TensorHandle hbias) {
    Tensor* W = (Tensor*)hW;
    Tensor* x = (Tensor*)hx;
    Tensor* bias = (Tensor*)hbias;
    if (W->dtype_tag == DT_F32 || x->dtype_tag == DT_F32 ||
        (bias && bias->dtype_tag == DT_F32)) {
        if (W->dtype_tag != x->dtype_tag ||
            (bias && bias->dtype_tag != W->dtype_tag)) tape_abort_mixed_dtype("tensor_linear");
        return tensor_linear_f32(hW, hx, hbias);
    }
    int m = W->shape[0], n = W->shape[1];
    int out_shape[] = {m};
    /* Zero-dim guard (see mm.c). cblas_dgemv rejects lda=0. */
    if (m == 0 || n == 0) {
        int rg0 = W->requires_grad || x->requires_grad || (bias && bias->requires_grad);
        Tensor* r = tape_zero_tensor(out_shape, 1, DT_F64, rg0);
        if (bias && m > 0) {
            for (int i = 0; i < m; i++)
                ((double*)r->data)[i] = ((double*)bias->data)[i];
        }
        return r;
    }
    double* out_data = arena_alloc(m * sizeof(double));
#ifdef __APPLE__
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
                W->data, n, x->data, 1, 0.0, out_data, 1);
#else
    for (int i = 0; i < m; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += ((double*)W->data)[i*n+j] * ((double*)x->data)[j];
        out_data[i] = s;
    }
#endif
    if (bias) {
#ifdef __APPLE__
        vDSP_vaddD(out_data, 1, bias->data, 1, out_data, 1, (vDSP_Length)m);
#else
        for (int i = 0; i < m; i++) out_data[i] += ((double*)bias->data)[i];
#endif
    }
    int rg = W->requires_grad || x->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor_arena(out_data, m, out_shape, 1, rg);
    if (rg) {
        TapeEntry* e = tape_append(OP_LINEAR, r, W, x, 0);
        LinearMeta* meta = arena_alloc(sizeof(LinearMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        memcpy(meta->x_vals, x->data, n * sizeof(double));
        meta->bias = bias;
        e->op_meta = meta;
    }
    return r;
}

static void tape_backward_linear(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;   /* W */
    Tensor* b = e->arg2;   /* x */
    LinearMeta* lm = (LinearMeta*)e->op_meta;
    int m_l = lm->m, n_l = lm->n;
    double* x_vals_l = lm->x_vals;
    ensure_grad(r);
    if (a->requires_grad) {
        ensure_grad(a);
        if (a->dtype_tag == DT_F32) {
            for (int ii = 0; ii < m_l; ii++)
                for (int jj = 0; jj < n_l; jj++)
                    tape_grad_add_d(a, ii*n_l+jj, tape_grad_load_d(r, ii) * x_vals_l[jj]);
        } else {
#ifdef __APPLE__
            cblas_dger(CblasRowMajor, m_l, n_l, 1.0,
                       r->grad, 1, x_vals_l, 1,
                       a->grad, n_l);
#else
            for (int ii = 0; ii < m_l; ii++)
                for (int jj = 0; jj < n_l; jj++)
                    tape_grad_add_d(a, ii*n_l+jj, tape_grad_load_d(r, ii) * x_vals_l[jj]);
#endif
        }
    }
    if (b && b->requires_grad) {
        ensure_grad(b);
        if (a->dtype_tag == DT_F32) {
            for (int jj = 0; jj < n_l; jj++) {
                double s = 0;
                for (int ii = 0; ii < m_l; ii++)
                    s += tape_load_d(a, ii*n_l+jj) * tape_grad_load_d(r, ii);
                tape_grad_add_d(b, jj, s);
            }
        } else {
#ifdef __APPLE__
            cblas_dgemv(CblasRowMajor, CblasTrans, m_l, n_l, 1.0,
                        a->data, n_l, r->grad, 1,
                        1.0, b->grad, 1);
#else
            for (int jj = 0; jj < n_l; jj++) {
                double s = 0;
                for (int ii = 0; ii < m_l; ii++) s += ((double*)a->data)[ii*n_l+jj] * tape_grad_load_d(r, ii);
                tape_grad_add_d(b, jj, s);
            }
#endif
        }
    }
    if (lm->bias && lm->bias->requires_grad) {
        ensure_grad(lm->bias);
        for (int ii = 0; ii < m_l; ii++)
            tape_grad_add_d(lm->bias, ii, tape_grad_load_d(r, ii));
    }
}

TAPE_REGISTER_OP(OP_LINEAR, tape_backward_linear)

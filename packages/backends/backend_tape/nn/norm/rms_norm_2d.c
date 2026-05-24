/* nn/norm/rms_norm_2d.c — row-wise RMS normalization (forward + backward).
 *
 * Formula (matches HF LlamaRMSNorm — no centering, no bias):
 *   variance_i = (1/n) * sum_j input[i, j]^2
 *   rstd_i     = 1 / sqrt(variance_i + eps)
 *   x_hat[i,j] = input[i, j] * rstd_i
 *   out[i, j]  = x_hat[i, j] * weight[j]
 *
 * Replaces the per-row 7-primitive chain in
 * `HfCommon.applyRmsNorm2dRaw` (narrow / mul / sum / mul_scalar /
 * add_scalar / sqrt / div / mul) with a single FFI call. The
 * decomposed chain emitted ~7*seqLen tape entries; this op emits one.
 *
 * F32 + F64 paths. x_hat + rstd cached in RmsNormMeta (always
 * double*) so backward reads uniformly.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_rms_norm_2d(TensorHandle h, TensorHandle hweight, double eps) {
    Tensor* t = (Tensor*)h;
    Tensor* weight = (Tensor*)hweight;
    if (t->dtype_tag != weight->dtype_tag)
        tape_abort_mixed_dtype("tensor_rms_norm_2d");
    int m = t->shape[0], n = t->shape[1];
    int shape[] = {m, n};
    int rg = t->requires_grad || weight->requires_grad;

    if (t->dtype_tag == DT_F32) {
        float* data = arena_alloc(m * n * sizeof(float));
        double* x_hat = malloc(m * n * sizeof(double));
        double* rstd = malloc(m * sizeof(double));
        const float* td = (const float*)t->data;
        const float* wd = (const float*)weight->data;
        for (int i = 0; i < m; i++) {
            double var = 0;
            for (int j = 0; j < n; j++) {
                double v = td[i*n+j];
                var += v * v;
            }
            var /= n;
            double inv_std = 1.0 / sqrt(var + eps);
            rstd[i] = inv_std;
            for (int j = 0; j < n; j++) {
                double xh = td[i*n+j] * inv_std;
                x_hat[i*n+j] = xh;
                data[i*n+j] = (float)(xh * wd[j]);
            }
        }
        Tensor* r = make_tensor_arena_f32(data, m * n, shape, 2, rg);
        if (rg) {
            RmsNormMeta* meta = arena_alloc(sizeof(RmsNormMeta));
            meta->weight = weight;
            meta->x_hat = x_hat; meta->rstd = rstd;
            meta->m = m; meta->n = n;
            TapeEntry* e = tape_append(OP_RMS_NORM_2D, r, t, NULL, 0);
            e->op_meta = meta;
        } else {
            free(x_hat); free(rstd);
        }
        return r;
    }

    double* data = malloc(m * n * sizeof(double));
    double* x_hat = malloc(m * n * sizeof(double));
    double* rstd = malloc(m * sizeof(double));
    const double* td = (const double*)t->data;
    const double* wd = (const double*)weight->data;
    for (int i = 0; i < m; i++) {
        double var = 0;
        for (int j = 0; j < n; j++) {
            double v = td[i*n+j];
            var += v * v;
        }
        var /= n;
        double inv_std = 1.0 / sqrt(var + eps);
        rstd[i] = inv_std;
        for (int j = 0; j < n; j++) {
            double xh = td[i*n+j] * inv_std;
            x_hat[i*n+j] = xh;
            data[i*n+j] = xh * wd[j];
        }
    }
    Tensor* r = make_tensor(data, shape, 2, rg);
    free(data);
    if (rg) {
        RmsNormMeta* meta = arena_alloc(sizeof(RmsNormMeta));
        meta->weight = weight;
        meta->x_hat = x_hat;
        meta->rstd = rstd;
        meta->m = m;
        meta->n = n;
        TapeEntry* e = tape_append(OP_RMS_NORM_2D, r, t, NULL, 0);
        e->op_meta = meta;
    } else {
        free(x_hat);
        free(rstd);
    }
    return r;
}

/* Backward.
 *   Let dout[i,j] = dL/dout[i,j], r_i = rstd_i, x_hat[i,j] = x[i,j] * r_i.
 *   d(weight)[j] = sum_i dout[i,j] * x_hat[i,j].
 *   d(x)[i,j]    = r_i * (dout_w[i,j] - x_hat[i,j] * (1/n) * sum_k (dout_w[i,k] * x_hat[i,k]))
 *                where dout_w[i,j] = dout[i,j] * weight[j].
 * Derivation in the commit body.
 */
static void tape_backward_rms_norm_2d(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    RmsNormMeta* meta = (RmsNormMeta*)e->op_meta;
    int mm = meta->m, nn = meta->n;
    ensure_grad(r);
    if (meta->weight && meta->weight->requires_grad) {
        ensure_grad(meta->weight);
        for (int j = 0; j < nn; j++) {
            double dw = 0;
            for (int i = 0; i < mm; i++) dw += ((double*)r->grad)[i*nn+j] * meta->x_hat[i*nn+j];
            ((double*)meta->weight->grad)[j] += dw;
        }
    }
    if (a && a->requires_grad) {
        ensure_grad(a);
        for (int i = 0; i < mm; i++) {
            double sum_dxhat_xhat = 0;
            for (int j = 0; j < nn; j++) {
                double dxhat = ((double*)r->grad)[i*nn+j] * tape_load_d(meta->weight, j);
                sum_dxhat_xhat += dxhat * meta->x_hat[i*nn+j];
            }
            double mean_dxhat_xhat = sum_dxhat_xhat / nn;
            for (int j = 0; j < nn; j++) {
                double dxhat = ((double*)r->grad)[i*nn+j] * tape_load_d(meta->weight, j);
                ((double*)a->grad)[i*nn+j] += meta->rstd[i] *
                    (dxhat - meta->x_hat[i*nn+j] * mean_dxhat_xhat);
            }
        }
    }
}

TAPE_REGISTER_OP(OP_RMS_NORM_2D, tape_backward_rms_norm_2d)

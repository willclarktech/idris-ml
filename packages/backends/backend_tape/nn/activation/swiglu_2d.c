/* nn/activation/swiglu_2d.c — fused silu(gate) * up (forward + backward).
 *
 * Forward:
 *   out[i,j] = silu(gate[i,j]) * up[i,j]
 *            = gate[i,j] * sigmoid(gate[i,j]) * up[i,j]
 *
 * Backward (let s = sigmoid(gate), silu(gate) = gate * s):
 *   d(silu)/d(gate) = s + gate * s * (1 - s) = s * (1 + gate * (1 - s))
 *   dL/d(gate)[i,j] = dout[i,j] * up[i,j] * s[i,j] * (1 + gate[i,j] * (1 - s[i,j]))
 *   dL/d(up)[i,j]   = dout[i,j] * gate[i,j] * s[i,j]
 *
 * Replaces the tsilu + tmul pair in HfLlama.applyMlp. The decomposed
 * chain emitted two tape entries (plus the sigmoid evaluated implicitly
 * inside silu); this op emits one and caches sigmoid(gate) so backward
 * skips the exp() re-evaluation.
 *
 * F32 + F64 paths. sig_g is always double* so backward is dtype-uniform.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

static inline double sigmoid_d(double x) {
    return 1.0 / (1.0 + exp(-x));
}

TensorHandle tensor_swiglu_2d(TensorHandle hgate, TensorHandle hup) {
    Tensor* g = (Tensor*)hgate;
    Tensor* u = (Tensor*)hup;
    if (g->dtype_tag != u->dtype_tag)
        tape_abort_mixed_dtype("tensor_swiglu_2d");
    int m = g->shape[0], n = g->shape[1];
    int shape[] = {m, n};
    int rg = g->requires_grad || u->requires_grad;

    if (g->dtype_tag == DT_F32) {
        float* data = arena_alloc(m * n * sizeof(float));
        double* sig_g = rg ? malloc(m * n * sizeof(double)) : NULL;
        const float* gd = (const float*)g->data;
        const float* ud = (const float*)u->data;
        for (int i = 0; i < m * n; i++) {
            double gv = gd[i];
            double s = sigmoid_d(gv);
            if (sig_g) sig_g[i] = s;
            data[i] = (float)(gv * s * (double)ud[i]);
        }
        Tensor* r = make_tensor_arena_f32(data, m * n, shape, 2, rg);
        if (rg) {
            SwiGluMeta* meta = arena_alloc(sizeof(SwiGluMeta));
            meta->sig_g = sig_g;
            meta->m = m; meta->n = n;
            TapeEntry* e = tape_append(OP_SWIGLU_2D, r, g, u, 0);
            e->op_meta = meta;
        }
        return r;
    }

    double* data = malloc(m * n * sizeof(double));
    double* sig_g = rg ? malloc(m * n * sizeof(double)) : NULL;
    const double* gd = (const double*)g->data;
    const double* ud = (const double*)u->data;
    for (int i = 0; i < m * n; i++) {
        double s = sigmoid_d(gd[i]);
        if (sig_g) sig_g[i] = s;
        data[i] = gd[i] * s * ud[i];
    }
    Tensor* r = make_tensor(data, shape, 2, rg);
    free(data);
    if (rg) {
        SwiGluMeta* meta = arena_alloc(sizeof(SwiGluMeta));
        meta->sig_g = sig_g;
        meta->m = m; meta->n = n;
        TapeEntry* e = tape_append(OP_SWIGLU_2D, r, g, u, 0);
        e->op_meta = meta;
    }
    return r;
}

static void tape_backward_swiglu_2d(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* g = e->arg1;
    Tensor* u = e->arg2;
    SwiGluMeta* meta = (SwiGluMeta*)e->op_meta;
    int N = meta->m * meta->n;
    ensure_grad(r);
    int gNeedsGrad = g && g->requires_grad;
    int uNeedsGrad = u && u->requires_grad;
    if (gNeedsGrad) ensure_grad(g);
    if (uNeedsGrad) ensure_grad(u);
    for (int i = 0; i < N; i++) {
        double dout = ((double*)r->grad)[i];
        double s = meta->sig_g[i];
        double gv = tape_load_d(g, i);
        if (gNeedsGrad) {
            double uv = tape_load_d(u, i);
            /* silu'(gate) = s * (1 + gate * (1 - s)) */
            double dsilu = s * (1.0 + gv * (1.0 - s));
            ((double*)g->grad)[i] += dout * uv * dsilu;
        }
        if (uNeedsGrad) {
            ((double*)u->grad)[i] += dout * gv * s;
        }
    }
}

TAPE_REGISTER_OP(OP_SWIGLU_2D, tape_backward_swiglu_2d)

/* nn/attention/cosine_similarity.c — row-wise cosine sim a[n,w] vs b[1,w]
 * (forward + backward).
 *
 * tape_load_d on both inputs covers F32 + F64. Compute
 * in double for numerical stability; narrow to float on output if F32.
 * Bias 1e-8 added to norms to avoid division-by-zero.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_cosine_similarity(TensorHandle ha, TensorHandle hb, int dim) {
    (void)dim;
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    if (a->rank == 2 && b->rank == 2) {
        if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_cosine_similarity");
        int n = a->shape[0], w = a->shape[1];
        int out_shape[] = {n};
        int rg = a->requires_grad || b->requires_grad;
        double bnorm2 = 0;
        for (int j = 0; j < w; j++) { double v = tape_load_d(b, j); bnorm2 += v * v; }
        double bnorm = sqrt(bnorm2) + 1e-8;
        Tensor* r;
        if (a->dtype_tag == DT_F32) {
            float* out = arena_alloc(n * sizeof(float));
            for (int i = 0; i < n; i++) {
                double dot = 0, anorm2 = 0;
                for (int j = 0; j < w; j++) {
                    double av = tape_load_d(a, i*w+j);
                    double bv = tape_load_d(b, j);
                    dot += av * bv;
                    anorm2 += av * av;
                }
                double anorm = sqrt(anorm2) + 1e-8;
                out[i] = (float)(dot / (anorm * bnorm));
            }
            r = make_tensor_arena_f32(out, n, out_shape, 1, rg);
        } else {
            double* out = calloc(n, sizeof(double));
            for (int i = 0; i < n; i++) {
                double dot = 0, anorm2 = 0;
                for (int j = 0; j < w; j++) {
                    double av = ((double*)a->data)[i*w+j];
                    double bv = ((double*)b->data)[j];
                    dot += av * bv;
                    anorm2 += av * av;
                }
                double anorm = sqrt(anorm2) + 1e-8;
                out[i] = dot / (anorm * bnorm);
            }
            r = make_tensor(out, out_shape, 1, rg);
            free(out);
        }
        if (r->requires_grad) tape_append(OP_COSINE_SIM, r, a, b, 0);
        return r;
    }
    return make_scalar(0, 0);
}

static void tape_backward_cosine_sim(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    if (a && a->rank == 2 && b && b->rank == 2) {
        int n_cs = a->shape[0], w_cs = a->shape[1];
        double bnorm2 = 0;
        for (int j = 0; j < w_cs; j++) { double v = tape_load_d(b, j); bnorm2 += v * v; }
        double bnorm = sqrt(bnorm2) + 1e-8;

        ensure_grad(a); ensure_grad(r);
        for (int ii = 0; ii < n_cs; ii++) {
            double anorm2 = 0;
            for (int j = 0; j < w_cs; j++) { double v = tape_load_d(a, ii*w_cs+j); anorm2 += v * v; }
            double anorm = sqrt(anorm2) + 1e-8;
            double cos_val = tape_load_d(r, ii);
            double g = ((double*)r->grad)[ii];
            for (int j = 0; j < w_cs; j++) {
                ((double*)a->grad)[ii*w_cs+j] += g * (tape_load_d(b, j) / (anorm * bnorm) - cos_val * tape_load_d(a, ii*w_cs+j) / (anorm2 + 1e-10));
            }
        }

        if (b->requires_grad) {
            ensure_grad(b);
            for (int ii = 0; ii < n_cs; ii++) {
                double anorm2 = 0;
                for (int j = 0; j < w_cs; j++) { double v = tape_load_d(a, ii*w_cs+j); anorm2 += v * v; }
                double anorm = sqrt(anorm2) + 1e-8;
                double cos_val = tape_load_d(r, ii);
                double g = ((double*)r->grad)[ii];
                for (int j = 0; j < w_cs; j++) {
                    ((double*)b->grad)[j] += g * (tape_load_d(a, ii*w_cs+j) / (anorm * bnorm) - cos_val * tape_load_d(b, j) / (bnorm2 + 1e-10));
                }
            }
        }
    }
}

TAPE_REGISTER_OP(OP_COSINE_SIM, tape_backward_cosine_sim)

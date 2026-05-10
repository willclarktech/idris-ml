/* conv/conv1d_circular.c — circular 1D convolution (forward + backward).
 *
 * Phase 1d.1.c. Used by NTM's content-addressing rotation step.
 * Input length n, kernel length k, pad=k/2. Wrap-around indexing
 * (the (i - pad + j + n) % n in both forward and backward).
 *
 *   out[i] = sum_j input[(i-pad+j) mod n] * kernel[k-1-j]
 *
 * Backward flips both contributions through the same wrap-around
 * indexing.
 */

#include <stdlib.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    if (input->dtype_tag != kernel->dtype_tag) tape_abort_mixed_dtype("tensor_conv1d_circular");
    int n = input->numel, k = kernel->numel, pad = k / 2;
    int shape[] = {n};
    int rg = input->requires_grad || kernel->requires_grad;
    if (input->dtype_tag == DT_F32) {
        float* out = arena_alloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            float s = 0;
            for (int j = 0; j < k; j++) {
                int idx = (i - pad + j + n) % n;
                s += ((float*)input->data)[idx] * ((float*)kernel->data)[k - 1 - j];
            }
            out[i] = s;
        }
        Tensor* r = make_tensor_arena_f32(out, n, shape, 1, rg);
        if (r->requires_grad) tape_append(OP_CONV1D_CIRC, r, input, kernel, 0);
        return r;
    }
    double* out = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < k; j++) {
            int idx = (i - pad + j + n) % n;
            s += ((double*)input->data)[idx] * ((double*)kernel->data)[k - 1 - j];
        }
        out[i] = s;
    }
    Tensor* r = make_tensor(out, shape, 1, rg);
    free(out);
    if (r->requires_grad) tape_append(OP_CONV1D_CIRC, r, input, kernel, 0);
    return r;
}

static void tape_backward_conv1d_circular(TapeEntry* e) {
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    Tensor* r = e->result;
    int n_cv = a->numel, k_cv = b->numel, pad_cv = k_cv / 2;
    ensure_grad(r);
    if (a->requires_grad) {
        ensure_grad(a);
        for (int ii = 0; ii < n_cv; ii++) {
            for (int j = 0; j < k_cv; j++) {
                int idx = (ii - pad_cv + j + n_cv) % n_cv;
                ((double*)a->grad)[idx] += ((double*)r->grad)[ii] * tape_load_d(b, k_cv - 1 - j);
            }
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        for (int ii = 0; ii < n_cv; ii++) {
            for (int j = 0; j < k_cv; j++) {
                int idx = (ii - pad_cv + j + n_cv) % n_cv;
                ((double*)b->grad)[k_cv - 1 - j] += ((double*)r->grad)[ii] * tape_load_d(a, idx);
            }
        }
    }
}

TAPE_REGISTER_OP(OP_CONV1D_CIRC, tape_backward_conv1d_circular)

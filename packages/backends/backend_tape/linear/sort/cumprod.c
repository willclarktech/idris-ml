/* linear/sort/cumprod.c — cumulative product along dim 0 (1D only).
 *
 * Forward: r[i] = prod(a[0..i]). Backward (safe form,
 * handles a[i] == 0):
 *   d_a[i] = (sum_{j>=i} d_r[j] * r[j]) / a[i]   when |a[i]| > 1e-30
 *   else exclusive-product recompute over the suffix.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    (void)dim;
    Tensor* t = (Tensor*)ht;
    int n = t->numel;
    int shape[] = {n};
    Tensor* r;
    if (t->dtype_tag == DT_F32) {
        float* out = arena_alloc(n * sizeof(float));
        float prod = 1.0f;
        for (int i = 0; i < n; i++) { prod *= ((float*)t->data)[i]; out[i] = prod; }
        r = make_tensor_arena_f32(out, n, shape, 1, t->requires_grad);
    } else {
        double* out = malloc(n * sizeof(double));
        double prod = 1.0;
        for (int i = 0; i < n; i++) { prod *= ((double*)t->data)[i]; out[i] = prod; }
        r = make_tensor(out, shape, 1, t->requires_grad);
        free(out);
    }
    if (r->requires_grad) tape_append(OP_CUMPROD, r, t, NULL, 0);
    return r;
}

static void tape_backward_cumprod(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    ensure_grad(r);
    if (a && a->requires_grad) {
        ensure_grad(a);
        int n = a->numel;
        double suffix_sum = 0.0;
        for (int i = n - 1; i >= 0; i--) {
            suffix_sum += tape_grad_load_d(r, i) * tape_load_d(r, i);
            double ai = tape_load_d(a, i);
            if (fabs(ai) > 1e-30) {
                tape_grad_add_d(a, i, suffix_sum / ai);
            } else {
                double partial = 0.0;
                for (int j = i; j < n; j++) {
                    double prod_excl = 1.0;
                    for (int k = 0; k <= j; k++) {
                        if (k != i) prod_excl *= tape_load_d(a, k);
                    }
                    partial += tape_grad_load_d(r, j) * prod_excl;
                }
                tape_grad_add_d(a, i, partial);
            }
        }
    }
}

TAPE_REGISTER_OP(OP_CUMPROD, tape_backward_cumprod)

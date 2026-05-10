/* nn/norm/group_norm.c — group normalization (forward only).
 *
 * Input treated as a flat 1D buffer of
 * length channels * spatial; output preserves the same shape. No
 * backward tape entry — torch/MLX handle group norm natively.
 *
 * Per-group statistics: compute (mean, var) over `chPerGroup * spatial`
 * elements, then normalize with rstd = 1/sqrt(var + eps) and apply the
 * per-channel affine (gamma, beta).
 */

#include <math.h>
#include <stdlib.h>
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    Tensor* input = (Tensor*)hinput;
    Tensor* gamma = (Tensor*)hgamma;
    Tensor* beta = (Tensor*)hbeta;
    int n = channels * spatial;
    int chPerGroup = channels / numGroups;
    int groupSize = chPerGroup * spatial;

    if (input->dtype_tag != gamma->dtype_tag || input->dtype_tag != beta->dtype_tag)
        tape_abort_mixed_dtype("tensor_group_norm");
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {n};
    void* out = is_f32 ? (void*)arena_alloc(n * sizeof(float))
                       : (void*)calloc(n, sizeof(double));
    for (int g = 0; g < numGroups; g++) {
        double mean = 0;
        int base = g * groupSize;
        for (int j = 0; j < groupSize; j++) mean += tape_load_d(input, base + j);
        mean /= groupSize;
        double var = 0;
        for (int j = 0; j < groupSize; j++) {
            double d = tape_load_d(input, base + j) - mean;
            var += d * d;
        }
        var /= groupSize;
        double rstd = 1.0 / sqrt(var + eps);
        for (int c = 0; c < chPerGroup; c++) {
            int absC = g * chPerGroup + c;
            double gc = tape_load_d(gamma, absC);
            double bc = tape_load_d(beta, absC);
            for (int s = 0; s < spatial; s++) {
                int idx = absC * spatial + s;
                double x_hat = (tape_load_d(input, idx) - mean) * rstd;
                double v = gc * x_hat + bc;
                if (is_f32) ((float*)out)[idx] = (float)v;
                else        ((double*)out)[idx] = v;
            }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, n, out_shape, 1, input->requires_grad || gamma->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 1, input->requires_grad || gamma->requires_grad); free(out); }
    /* No backward tape entry — torch/MLX handle group norm natively. */
    return r;
}

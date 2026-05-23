/* core/scalar/clamp.c — element-wise two-sided clamp by scalars.
 *
 * Forward: r[i] = min(max(t[i], lo), hi). Inference-only (no tape
 * entry); the BitNet activation quantization path is the primary
 * user, which is no-grad by construction. A differentiable variant
 * would mirror `tensor_clamp_min`'s pass-through-in-range backward
 * — file as follow-up if a training path needs it.
 */

#include <stdlib.h>
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

static TensorHandle tensor_clamp_f32(TensorHandle ha, double lo, double hi) {
    Tensor* a = (Tensor*)ha;
    int n = a->numel;
    float flo = (float)lo;
    float fhi = (float)hi;
    float* data = arena_alloc(n * sizeof(float));
    const float* src = (const float*)a->data;
    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (v < flo) v = flo;
        if (v > fhi) v = fhi;
        data[i] = v;
    }
    Tensor* r = (n == 1) ? make_scalar_f32((double)data[0], 0)
                         : make_tensor_arena_f32(data, n, a->shape, a->rank, 0);
    return r;
}

TensorHandle tensor_clamp(TensorHandle ha, double lo, double hi) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return tensor_clamp_f32(ha, lo, hi);
    int n = a->numel;
    double* data = malloc(n * sizeof(double));
    const double* src = (const double*)a->data;
    for (int i = 0; i < n; i++) {
        double v = src[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        data[i] = v;
    }
    Tensor* r = (n == 1) ? make_scalar(data[0], 0)
                         : make_tensor(data, a->shape, a->rank, 0);
    free(data);
    return r;
}

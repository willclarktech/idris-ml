/* core/scalar/round.c — element-wise round-to-nearest-even.
 *
 * Inference-only (no tape entry). C's `rint()` follows the current
 * rounding mode (default banker's rounding = round-half-to-even),
 * matching `torch.round` and `mx::round`. The BitNet activation
 * quantization path is the primary user.
 */

#include <math.h>
#include <stdlib.h>
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

static TensorHandle tensor_round_f32(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    int n = a->numel;
    float* data = arena_alloc(n * sizeof(float));
    const float* src = (const float*)a->data;
    for (int i = 0; i < n; i++) data[i] = rintf(src[i]);
    Tensor* r = (n == 1) ? make_scalar_f32((double)data[0], 0)
                         : make_tensor_arena_f32(data, n, a->shape, a->rank, 0);
    return r;
}

TensorHandle tensor_round(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return tensor_round_f32(ha);
    int n = a->numel;
    double* data = malloc(n * sizeof(double));
    const double* src = (const double*)a->data;
    for (int i = 0; i < n; i++) data[i] = rint(src[i]);
    Tensor* r = (n == 1) ? make_scalar(data[0], 0)
                         : make_tensor(data, a->shape, a->rank, 0);
    free(data);
    return r;
}

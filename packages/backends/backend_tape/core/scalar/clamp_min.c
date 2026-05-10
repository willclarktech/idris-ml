/* core/scalar/clamp_min.c — element-wise clamp_min (forward + backward).
 *
 * Forward: r[i] = max(x[i], min_val). Backward: gradient
 * passes through where x > min_val, zero where clamped.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

static TensorHandle tensor_clamp_min_f32(TensorHandle ha, double min_val) {
    Tensor* a = (Tensor*)ha;
    int n = a->numel;
    float mv = (float)min_val;
    float* data = arena_alloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float v = ((float*)a->data)[i];
        data[i] = v > mv ? v : mv;
    }
    Tensor* r = (n == 1) ? make_scalar_f32((double)data[0], a->requires_grad)
                         : make_tensor_arena_f32(data, n, a->shape, a->rank, a->requires_grad);
    if (r->requires_grad) tape_append(OP_CLAMP_MIN, r, a, NULL, min_val);
    return r;
}

TensorHandle tensor_clamp_min(TensorHandle ha, double min_val) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return tensor_clamp_min_f32(ha, min_val);
    int n = a->numel;
    double* data = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) data[i] = fmax(((double*)a->data)[i], min_val);
    Tensor* r;
    if (n == 1) {
        r = make_scalar(data[0], a->requires_grad);
    } else {
        r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    }
    free(data);
    if (r->requires_grad) tape_append(OP_CLAMP_MIN, r, a, NULL, min_val);
    return r;
}

static void tape_backward_clamp_min(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    double min_val = e->scalar_arg;
    if (a) {
        ensure_grad(a);
        ensure_grad(r);
        for (int j = 0; j < a->numel; j++)
            ((double*)a->grad)[j] += (tape_load_d(a, j) > min_val) ? ((double*)r->grad)[j] : 0.0;
    }
}

TAPE_REGISTER_OP(OP_CLAMP_MIN, tape_backward_clamp_min)

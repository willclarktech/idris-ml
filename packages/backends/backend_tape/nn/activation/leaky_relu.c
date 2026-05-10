/* nn/activation/leaky_relu.c — leaky-relu (forward + backward).
 *
 * Phase 1c.2. Forward: max(alpha*x, x) (alpha stored in scalar_arg).
 * Backward: d_x = 1 (x >= 0) or alpha (x < 0). tape_load_d covers F32+F64.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

static TensorHandle tensor_leaky_relu_f32(TensorHandle ha, double alpha) {
    Tensor* a = (Tensor*)ha;
    float af = (float)alpha;
    if (a->numel == 1) {
        float x = ((float*)a->data)[0];
        Tensor* r = make_scalar_f32((double)(x >= 0 ? x : af * x), a->requires_grad);
        if (a->requires_grad) tape_append(OP_LEAKY_RELU, r, a, NULL, alpha);
        return r;
    }
    float* data = arena_alloc(a->numel * sizeof(float));
    for (int i = 0; i < a->numel; i++) {
        float x = ((float*)a->data)[i];
        data[i] = x >= 0 ? x : af * x;
    }
    Tensor* r = make_tensor_arena_f32(data, a->numel, a->shape, a->rank, a->requires_grad);
    if (a->requires_grad) tape_append(OP_LEAKY_RELU, r, a, NULL, alpha);
    return r;
}

TensorHandle tensor_leaky_relu(TensorHandle ha, double alpha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return tensor_leaky_relu_f32(ha, alpha);
    if (a->numel == 1) {
        double x = ((double*)a->data)[0];
        Tensor* r = make_scalar(x >= 0 ? x : alpha * x, a->requires_grad);
        if (a->requires_grad) tape_append(OP_LEAKY_RELU, r, a, NULL, alpha);
        return r;
    }
    double* data = malloc(a->numel * sizeof(double));
    for (int i = 0; i < a->numel; i++) {
        double x = ((double*)a->data)[i];
        data[i] = x >= 0 ? x : alpha * x;
    }
    Tensor* r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    free(data);
    if (a->requires_grad) tape_append(OP_LEAKY_RELU, r, a, NULL, alpha);
    return r;
}

static void tape_backward_leaky_relu(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    double alpha = e->scalar_arg;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            ((double*)a->grad)[j] += ((double*)r->grad)[j] * (tape_load_d(a, j) >= 0 ? 1.0 : alpha);
    }
}

TAPE_REGISTER_OP(OP_LEAKY_RELU, tape_backward_leaky_relu)

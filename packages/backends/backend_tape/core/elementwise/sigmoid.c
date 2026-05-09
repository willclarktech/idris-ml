/* core/elementwise/sigmoid.c — sigmoid (forward + backward).
 *
 * Phase 1a.8. d sigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x))
 *                            = r * (1 - r).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_sigmoid_d(double x) { return 1.0 / (1.0 + exp(-x)); }
static float  fn_sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }

TensorHandle tensor_sigmoid(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_SIGMOID, fn_sigmoid_f32);
    return unop_elementwise(ha, OP_SIGMOID, fn_sigmoid_d);
}

static void tape_backward_sigmoid(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++) {
            double s = tape_load_d(r, j);
            ((double*)a->grad)[j] += ((double*)r->grad)[j] * s * (1.0 - s);
        }
    }
}

TAPE_REGISTER_OP(OP_SIGMOID, tape_backward_sigmoid)

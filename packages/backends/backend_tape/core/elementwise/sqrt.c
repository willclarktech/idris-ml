/* core/elementwise/sqrt.c — element-wise square root (forward + backward).
 *
 * Phase 1a.6. d(sqrt(x))/dx = 1 / (2 * sqrt(x)) = 1 / (2 * r).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_sqrt_d(double x) { return sqrt(x); }
static float  fn_sqrt_f32(float x) { return sqrtf(x); }

TensorHandle tensor_sqrt(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_SQRT, fn_sqrt_f32);
    return unop_elementwise(ha, OP_SQRT, fn_sqrt_d);
}

static void tape_backward_sqrt(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            ((double*)a->grad)[j] += ((double*)r->grad)[j] / (2.0 * tape_load_d(r, j));
    }
}

TAPE_REGISTER_OP(OP_SQRT, tape_backward_sqrt)

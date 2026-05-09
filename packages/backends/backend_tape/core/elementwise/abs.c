/* core/elementwise/abs.c — element-wise absolute value (forward + backward).
 *
 * Phase 1a.6. d|x|/dx = sign(x) (discontinuous at 0; we pick +1 for x=0).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_abs_d(double x) { return fabs(x); }
static float  fn_abs_f32(float x) { return fabsf(x); }

TensorHandle tensor_abs(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_ABS, fn_abs_f32);
    return unop_elementwise(ha, OP_ABS, fn_abs_d);
}

static void tape_backward_abs(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            ((double*)a->grad)[j] += ((double*)r->grad)[j] * (tape_load_d(a, j) >= 0 ? 1.0 : -1.0);
    }
}

TAPE_REGISTER_OP(OP_ABS, tape_backward_abs)

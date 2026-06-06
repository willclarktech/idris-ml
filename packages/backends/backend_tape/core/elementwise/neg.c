/* core/elementwise/neg.c — unary negation (forward + backward).
 *
 * d(-x)/dx = -1.
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_neg(double x) { return -x; }
static float  fn_neg_f32(float x) { return -x; }

TensorHandle tensor_neg(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_NEG, fn_neg_f32);
    return unop_elementwise(ha, OP_NEG, fn_neg);
}

static void tape_backward_neg(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++) tape_grad_add_d(a, j, -(tape_grad_load_d(r, j)));
    }
}

TAPE_REGISTER_OP(OP_NEG, tape_backward_neg)

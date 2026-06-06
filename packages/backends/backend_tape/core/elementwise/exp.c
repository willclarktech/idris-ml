/* core/elementwise/exp.c — element-wise exponential (forward + backward).
 *
 * d(exp(x))/dx = exp(x) = r (the forward output).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_exp_d(double x) { return exp(x); }
static float  fn_exp_f32(float x) { return expf(x); }

TensorHandle tensor_exp(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_EXP, fn_exp_f32);
    return unop_elementwise(ha, OP_EXP, fn_exp_d);
}

static void tape_backward_exp(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            tape_grad_add_d(a, j, tape_grad_load_d(r, j) * tape_load_d(r, j));
    }
}

TAPE_REGISTER_OP(OP_EXP, tape_backward_exp)

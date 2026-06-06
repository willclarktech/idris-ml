/* core/elementwise/tanh.c — hyperbolic tangent (forward + backward).
 *
 * d tanh(x)/dx = 1 - tanh(x)^2 = 1 - r^2.
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_tanh_d(double x) { return tanh(x); }
static float  fn_tanh_f32(float x) { return tanhf(x); }

TensorHandle tensor_tanh(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_TANH, fn_tanh_f32);
    return unop_elementwise(ha, OP_TANH, fn_tanh_d);
}

static void tape_backward_tanh(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++) {
            double t = tape_load_d(r, j);
            tape_grad_add_d(a, j, tape_grad_load_d(r, j) * (1.0 - t * t));
        }
    }
}

TAPE_REGISTER_OP(OP_TANH, tape_backward_tanh)

/* core/elementwise/log.c — element-wise natural log (forward + backward).
 *
 * Phase 1a.6. d(log(x))/dx = 1/x (undefined for x<=0; we trust callers).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_log_d(double x) { return log(x); }
static float  fn_log_f32(float x) { return logf(x); }

TensorHandle tensor_log(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_LOG, fn_log_f32);
    return unop_elementwise(ha, OP_LOG, fn_log_d);
}

static void tape_backward_log(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            ((double*)a->grad)[j] += ((double*)r->grad)[j] / tape_load_d(a, j);
    }
}

TAPE_REGISTER_OP(OP_LOG, tape_backward_log)

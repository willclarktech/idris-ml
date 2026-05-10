/* nn/loss/bce_with_logits.c — binary cross-entropy with logits.
 *
 * Phase 1c.6. Stable formulation: max(p,0) - p*y + log(1+exp(-|p|)).
 * Backward: d_input[i] = (sigmoid(p_i) - y_i) / n.
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_bce_with_logits(TensorHandle hinput, TensorHandle htarget) {
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_bce_with_logits");
    int n = input->numel;
    double loss = 0;
    for (int i = 0; i < n; i++) {
        double p = tape_load_d(input, i), y = tape_load_d(target, i);
        double max_p = p > 0 ? p : 0;
        loss += max_p - p * y + log(1.0 + exp(-fabs(p)));
    }
    loss /= n;
    Tensor* r = (input->dtype_tag == DT_F32)
                  ? make_scalar_f32(loss, input->requires_grad)
                  : make_scalar(loss, input->requires_grad);
    if (r->requires_grad) tape_append(OP_BCE_WITH_LOGITS, r, input, target, 0);
    return r;
}

static void tape_backward_bce_with_logits(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    if (a) {
        ensure_grad(a);
        int n_bce = a->numel;
        for (int j = 0; j < n_bce; j++) {
            double sig = 1.0 / (1.0 + exp(-tape_load_d(a, j)));
            ((double*)a->grad)[j] += ((double*)r->grad)[0] * (sig - tape_load_d(b, j)) / n_bce;
        }
    }
}

TAPE_REGISTER_OP(OP_BCE_WITH_LOGITS, tape_backward_bce_with_logits)

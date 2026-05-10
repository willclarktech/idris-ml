/* nn/loss/mse_loss.c — forward-only MSE.
 *
 * loss = mean((input - target)^2). No backward — see
 * the cross-entropy companion file for the same rationale.
 */

#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_mse_loss(TensorHandle hinput, TensorHandle htarget) {
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_mse_loss");
    double loss = 0;
    for (int i = 0; i < input->numel; i++) {
        double d = tape_load_d(input, i) - tape_load_d(target, i);
        loss += d * d;
    }
    double mean = loss / input->numel;
    return (input->dtype_tag == DT_F32) ? make_scalar_f32(mean, 0) : make_scalar(mean, 0);
}

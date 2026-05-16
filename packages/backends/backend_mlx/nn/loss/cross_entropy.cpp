/* tensor_cross_entropy for the mlx backend.
 *
 * Cross-entropy with soft labels: CE = -mean(target * log_softmax(input)).
 * Decomposed into primitives so each step records its own tape entry —
 * backward flows through replay-based vjp.
 *
 * Matches the tape backend's choice of dim=0 for log_softmax for
 * cross-backend consistency. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_log_softmax(TensorHandle h, int dim);
extern "C" TensorHandle tensor_mul(TensorHandle ha, TensorHandle hb);
extern "C" TensorHandle tensor_neg(TensorHandle h);
extern "C" TensorHandle tensor_mean(TensorHandle h);

extern "C" TensorHandle tensor_cross_entropy(TensorHandle hinput, TensorHandle htarget) {
    TensorHandle ls = tensor_log_softmax(hinput, 0);
    TensorHandle prod = tensor_mul(htarget, ls);
    TensorHandle neg = tensor_neg(prod);
    return tensor_mean(neg);
}

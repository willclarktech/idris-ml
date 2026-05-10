/* linear/shape/unsqueeze.c — insert a size-1 dimension.
 *
 * For rank-1 input, delegates to
 * tensor_reshape which both shares storage and emits an OP_RESHAPE
 * tape entry (so backward flows via reshape's grad-passthrough).
 * For other ranks the simplified semantics return a clone — proper
 * arbitrary-dim insertion is future work.
 */

#include "../../tensor.h"
#include "../../../backend.h"

TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
    (void)dim;
    Tensor* t = (Tensor*)h;
    if (t->rank == 1) {
        int shape[] = {1, t->numel};
        return tensor_reshape(h, shape, 2);
    }
    return tensor_clone(h);
}

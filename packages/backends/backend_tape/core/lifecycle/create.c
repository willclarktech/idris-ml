/* core/lifecycle/create.c — Allocate an n-dim tensor from a host buffer.
 *
 * Arena-allocated (intermediate); becomes persistent only via
 * tensor_create_param_* (a separate lifecycle path).
 */

#include "../../tape.h"
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    Tensor* t = make_tensor(data, shape, rank, requires_grad);
    if (requires_grad) {
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
    return t;
}

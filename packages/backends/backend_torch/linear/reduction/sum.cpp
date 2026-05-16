/* tensor_sum / tensor_sum_dim for the torch backend.
 *
 * libtorch's autograd graph carries the backward; no tape entry needed. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sum(TensorHandle h) {
    return from_tensor(to_tensor(h)->sum());
}

extern "C" TensorHandle tensor_sum_dim(TensorHandle h, int dim, int keepdim) {
    return from_tensor(to_tensor(h)->sum(dim, keepdim != 0));
}

/* tensor_concat_2d_axis1 for the torch backend.
 * A: [m, n], B: [m, k] -> [m, n+k] along axis 1. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_concat_2d_axis1(TensorHandle A, TensorHandle B) {
    return from_tensor(torch::cat({*to_tensor(A), *to_tensor(B)}, 1));
}

/* tensor_dot for the torch backend (1D dot product). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_dot(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::dot(*to_tensor(a), *to_tensor(b)));
}

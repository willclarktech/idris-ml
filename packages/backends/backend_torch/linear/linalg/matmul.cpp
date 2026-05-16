/* tensor_matmul for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_matmul(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::matmul(*to_tensor(a), *to_tensor(b)));
}

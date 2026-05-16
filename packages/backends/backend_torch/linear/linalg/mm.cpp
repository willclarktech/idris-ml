/* tensor_mm for the torch backend (matrix-matrix, 2D). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mm(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::mm(*to_tensor(a), *to_tensor(b)));
}

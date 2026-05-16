/* tensor_mv for the torch backend (matrix-vector product). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mv(TensorHandle mat, TensorHandle vec) {
    return from_tensor(torch::mv(*to_tensor(mat), *to_tensor(vec)));
}

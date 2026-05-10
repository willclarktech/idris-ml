/* tensor_sub for the torch backend. See add.cpp for the libtorch-vs-tape
 * autograd contrast. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sub(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::sub(*to_tensor(a), *to_tensor(b)));
}

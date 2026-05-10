/* tensor_mul for the torch backend. See add.cpp for the libtorch-vs-tape
 * autograd contrast. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mul(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::mul(*to_tensor(a), *to_tensor(b)));
}

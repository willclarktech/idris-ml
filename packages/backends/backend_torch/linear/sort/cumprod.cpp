/* tensor_cumprod for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    return from_tensor(torch::cumprod(*to_tensor(ht), dim));
}

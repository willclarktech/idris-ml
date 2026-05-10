/* tensor_pow for the torch backend (elementwise base ^ exp). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_pow(TensorHandle base, TensorHandle exp) {
    return from_tensor(torch::pow(*to_tensor(base), *to_tensor(exp)));
}

/* tensor_softplus for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_softplus(TensorHandle h) {
    return from_tensor(torch::softplus(*to_tensor(h)));
}

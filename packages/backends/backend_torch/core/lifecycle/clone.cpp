/* tensor_clone for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_clone(TensorHandle h) {
    return from_tensor(to_tensor(h)->clone());
}

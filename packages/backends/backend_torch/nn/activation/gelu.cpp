/* tensor_gelu for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_gelu(TensorHandle h) {
    return from_tensor(torch::gelu(*to_tensor(h)));
}

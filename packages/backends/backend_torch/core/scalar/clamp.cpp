/* tensor_clamp for the torch backend. Two-sided scalar clamp. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_clamp(TensorHandle h, double lo, double hi) {
    return from_tensor(at::clamp(*to_tensor(h), lo, hi));
}

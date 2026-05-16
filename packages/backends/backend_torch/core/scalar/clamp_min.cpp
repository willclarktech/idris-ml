/* tensor_clamp_min for the torch backend. clamp_min is the autograd-
 * aware lower-bound clamp; backward zeros gradient at clamped indices. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_clamp_min(TensorHandle h, double min_val) {
    return from_tensor(torch::clamp_min(*to_tensor(h), min_val));
}

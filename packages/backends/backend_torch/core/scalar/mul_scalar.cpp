/* tensor_mul_scalar for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mul_scalar(TensorHandle h, double s) {
    return from_tensor(*to_tensor(h) * s);
}

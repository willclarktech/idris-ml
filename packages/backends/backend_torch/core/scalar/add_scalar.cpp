/* tensor_add_scalar for the torch backend. Operator+(Tensor, Scalar)
 * triggers libtorch's autograd-aware add-with-broadcasted-scalar path. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_add_scalar(TensorHandle h, double s) {
    return from_tensor(*to_tensor(h) + s);
}

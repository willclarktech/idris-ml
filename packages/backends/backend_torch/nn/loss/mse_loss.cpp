/* tensor_mse_loss for the torch backend. Mean-reduced (matches torch's
 * default and tape's convention). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mse_loss(TensorHandle input, TensorHandle target) {
    return from_tensor(torch::mse_loss(*to_tensor(input), *to_tensor(target)));
}

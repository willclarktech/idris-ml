/* tensor_bce_with_logits for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_bce_with_logits(TensorHandle input, TensorHandle target) {
    return from_tensor(torch::nn::functional::binary_cross_entropy_with_logits(
        *to_tensor(input), *to_tensor(target)));
}

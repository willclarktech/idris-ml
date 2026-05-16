/* tensor_linear (Wx + b) and tensor_linear_2d (X @ W^T + b) for the
 * torch backend.
 *
 * The 1D form uses torch::mv to match tape's 1D linear semantics.
 * The 2D form uses torch::nn::functional::linear which expects the
 * Y = X W^T + b layout (X: [B, i], W: [o, i], bias: [o] -> Y: [B, o]). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_linear(TensorHandle W, TensorHandle x, TensorHandle bias) {
    auto result = torch::mv(*to_tensor(W), *to_tensor(x));
    if (bias) result = result + *to_tensor(bias);
    return from_tensor(result);
}

extern "C" TensorHandle tensor_linear_2d(TensorHandle W, TensorHandle X, TensorHandle bias) {
    auto result = torch::nn::functional::linear(*to_tensor(X), *to_tensor(W),
                                                bias ? *to_tensor(bias) : torch::Tensor{});
    return from_tensor(result);
}

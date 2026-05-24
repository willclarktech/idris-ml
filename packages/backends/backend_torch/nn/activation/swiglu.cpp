/* tensor_swiglu_2d for the torch backend.
 *
 *   out = silu(gate) * up
 *
 * Composed via at::silu + at::mul. libtorch's autograd handles the
 * backward pass for both inputs automatically. Replaces the tsilu +
 * tmul pair in HfLlama.applyMlp with one FFI call. On torch-mps the
 * kernel set fuses into a smaller MPSGraph subgraph.
 */
#include "../../tensor.h"

extern "C" TensorHandle tensor_swiglu_2d(TensorHandle gate, TensorHandle up) {
    auto& g = *to_tensor(gate);
    auto& u = *to_tensor(up);
    auto out = at::mul(at::silu(g), u);
    return from_tensor(std::move(out));
}

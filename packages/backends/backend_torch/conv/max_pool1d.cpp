/* tensor_max_pool1d for the torch backend. */
#include "../tensor.h"

extern "C" TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    auto& inp = *to_tensor(hinput);
    auto inp_3d = inp.unsqueeze(0);
    auto out = torch::max_pool1d(inp_3d, {kL}, {stride});
    return from_tensor(out.squeeze(0));
}

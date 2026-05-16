/* tensor_gather for the torch backend. Indices are coerced to kLong
 * (torch::index_select requires int64). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    (void)n;
    auto& inp = *to_tensor(hinput);
    auto& idx = *to_tensor(hindex);
    auto idx_long = idx.to(torch::kLong);
    return from_tensor(torch::index_select(inp, 0, idx_long));
}

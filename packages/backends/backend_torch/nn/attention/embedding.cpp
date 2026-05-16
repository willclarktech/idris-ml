/* tensor_embedding for the torch backend.
 *
 * Indices are coerced to kLong (torch::embedding requires int64).
 * The result is flattened to [n * embedDim] so the FFI consumer sees
 * a 1D buffer — the Idris layer reshapes back as needed. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    (void)n; (void)embedDim;
    auto& weight = *to_tensor(hweight);
    auto& indices = *to_tensor(hindices);
    auto idx_long = indices.to(torch::kLong);
    auto out = torch::embedding(weight, idx_long);
    return from_tensor(out.reshape({-1}));
}

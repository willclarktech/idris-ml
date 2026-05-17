/* tensor_embedding for the torch backend.
 *
 * Indices are coerced to kLong (torch::embedding requires int64) AND
 * moved to the weight's device — the cast alone preserves device, so
 * a CPU-allocated index tensor would clash with an MPS-resident
 * weight at the MPS `index_select` placeholder check. Surfaced via
 * the BF16-on-MPS Llama bring-up 2026-05-28 (the F32 lane happened to
 * land both args on MPS via earlier device-aware ops; the BF16 lane
 * exposed the device mismatch). The result is flattened to
 * [n * embedDim] so the FFI consumer sees a 1D buffer — the Idris
 * layer reshapes back as needed. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    (void)n; (void)embedDim;
    auto& weight = *to_tensor(hweight);
    auto& indices = *to_tensor(hindices);
    auto idx_long = indices.to(torch::TensorOptions().dtype(torch::kLong).device(weight.device()));
    auto out = torch::embedding(weight, idx_long);
    return from_tensor(out.reshape({-1}));
}

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
 * layer reshapes back as needed.
 *
 * Cast is guarded so the common (indices already int64 + on weight's
 * device) case skips an `.to()` no-op submission; on MPS that no-op
 * still queues an MTLCommandBuffer per embedding lookup, contributing
 * to the per-op submission overhead tracked under TODO #393. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    (void)n; (void)embedDim;
    auto& weight = *to_tensor(hweight);
    auto& indices = *to_tensor(hindices);
    auto idx_long = (indices.scalar_type() == torch::kLong &&
                     indices.device() == weight.device())
                    ? indices
                    : indices.to(torch::TensorOptions()
                                 .dtype(torch::kLong)
                                 .device(weight.device()));
    auto out = torch::embedding(weight, idx_long);
    return from_tensor(out.reshape({-1}));
}

/* 2D-returning variant: keeps the [n, embedDim] shape that
 * torch::embedding produces natively. Callers in HfLlama / HfBert /
 * HfGpt2 / HfBitNet / Layer/Transformer drop their trailing
 * primReshape2d when they use this. */
extern "C" TensorHandle tensor_embedding_2d(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    (void)n; (void)embedDim;
    auto& weight = *to_tensor(hweight);
    auto& indices = *to_tensor(hindices);
    auto idx_long = (indices.scalar_type() == torch::kLong &&
                     indices.device() == weight.device())
                    ? indices
                    : indices.to(torch::TensorOptions()
                                 .dtype(torch::kLong)
                                 .device(weight.device()));
    return from_tensor(torch::embedding(weight, idx_long));
}

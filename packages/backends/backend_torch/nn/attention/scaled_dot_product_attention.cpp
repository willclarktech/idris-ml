/* tensor_sdpa_2d for the torch backend (TODO #399 Commit B).
 *
 * Wraps `at::scaled_dot_product_attention`, which on torch-mps routes
 * to MPSGraph's fused attention kernel (one MTLCommandBuffer for the
 * whole QK^T → scale → mask → softmax → V composition). Replaces the
 * Idris-side per-head attention loop (~10K ops/forward on Llama-3.2-1B)
 * with one fused libtorch call. GQA (numHeads ≠ numKvHeads) goes
 * through libtorch's `enable_gqa` flag.
 *
 * I/O is 2D-flat so callers don't pay multiplicative-Nat elaboration
 * cost in their type signatures:
 *   Q : [seq, numHeads   * headDim]
 *   K : [seq, numKvHeads * headDim]
 *   V : [seq, numKvHeads * headDim]
 *   out [seq, numHeads * headDim]
 *
 * Internal: reshape + transpose to the [..., L, E] layout SDPA expects
 * ([numHeads, seq, headDim] for Q; [numKvHeads, seq, headDim] for K/V),
 * call SDPA, transpose + contiguous + reshape back. Each reshape /
 * transpose on torch is metadata-only when contiguous (and contiguous
 * for the final view forces one materialization — necessary so the
 * output buffer the FFI hands back is row-major).
 *
 * Caller's responsibility: Q and K must already have RoPE applied
 * per-head before this call (the per-head RoPE loop in
 * applyAttention's caller path stays intact under this Commit). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sdpa_2d(
    TensorHandle hq, TensorHandle hk, TensorHandle hv,
    int numHeads, int numKvHeads, int headDim, int isCausal) {
    auto& q = *to_tensor(hq);  // [seq, numHeads   * headDim]
    auto& k = *to_tensor(hk);  // [seq, numKvHeads * headDim]
    auto& v = *to_tensor(hv);  // [seq, numKvHeads * headDim]
    int64_t seq = q.size(0);
    // [seq, nH * hd] -> [seq, nH, hd] -> [nH, seq, hd]
    auto q3 = q.view({seq, (int64_t)numHeads,   (int64_t)headDim}).transpose(0, 1);
    auto k3 = k.view({seq, (int64_t)numKvHeads, (int64_t)headDim}).transpose(0, 1);
    auto v3 = v.view({seq, (int64_t)numKvHeads, (int64_t)headDim}).transpose(0, 1);
    auto out3 = at::scaled_dot_product_attention(
                    q3, k3, v3,
                    /*attn_mask=*/std::nullopt,
                    /*dropout_p=*/0.0,
                    /*is_causal=*/isCausal != 0,
                    /*scale=*/std::nullopt,
                    /*enable_gqa=*/numHeads != numKvHeads);
    // [nH, seq, hd] -> [seq, nH, hd] -> [seq, nH * hd]
    auto out2 = out3.transpose(0, 1).contiguous()
                    .view({seq, (int64_t)numHeads * (int64_t)headDim});
    return from_tensor(std::move(out2));
}

/* tensor_sdpa_2d for the torch backend.
 *
 * Wraps `at::scaled_dot_product_attention`, which on torch-mps routes
 * to MPSGraph's fused attention kernel (one MTLCommandBuffer for the
 * whole QK^T → scale → mask → softmax → V composition). Replaces the
 * Idris-side per-head attention loop (~10K ops/forward on Llama-3.2-1B)
 * with one fused libtorch call. GQA (numHeads ≠ numKvHeads) goes
 * through libtorch's `enable_gqa` flag.
 *
 * I/O is 2D-flat so callers don't pay multiplicative-Nat elaboration
 * cost in their type signatures. **Q and KV may have different
 * sequence lengths** — that's the cache-aware decode path: Q is a
 * single new token (q_seq=1), K/V cover the full history including
 * cache (kv_seq=cache_len+1). For prefill / training, q_seq == kv_seq.
 *   Q : [q_seq,  numHeads   * headDim]
 *   K : [kv_seq, numKvHeads * headDim]
 *   V : [kv_seq, numKvHeads * headDim]
 *   out [q_seq, numHeads * headDim]
 *
 * Internal: reshape + transpose to the [..., L, E] layout SDPA expects
 * ([numHeads, seq, headDim] for Q; [numKvHeads, seq, headDim] for K/V),
 * call SDPA, transpose + contiguous + reshape back. Each reshape /
 * transpose on torch is metadata-only when contiguous (and contiguous
 * for the final view forces one materialization — necessary so the
 * output buffer the FFI hands back is row-major).
 *
 * Causal mask under asymmetric (q_seq < kv_seq): `is_causal=true`
 * aligns the mask to the lower-right corner of the [q_seq, kv_seq]
 * grid — query position i sees K positions [0 .. kv_seq - q_seq + i].
 * This is exactly the cache-aware semantics: query position
 * cache_len + i (absolute) sees all KV positions [0..cache_len + i].
 *
 * Caller's responsibility: Q and K must already have RoPE applied
 * per-head before this call. For cache-aware decode, Q gets RoPE at
 * position cache_len + i and K_new gets RoPE at the same offset
 * before being concatenated into the cache. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sdpa_2d(
    TensorHandle hq, TensorHandle hk, TensorHandle hv,
    int numHeads, int numKvHeads, int headDim, int isCausal) {
    auto& q = *to_tensor(hq);  // [q_seq,  numHeads   * headDim]
    auto& k = *to_tensor(hk);  // [kv_seq, numKvHeads * headDim]
    auto& v = *to_tensor(hv);  // [kv_seq, numKvHeads * headDim]
    int64_t q_seq  = q.size(0);
    int64_t kv_seq = k.size(0);
    // [q_seq, nH * hd] -> [q_seq, nH, hd] -> [nH, q_seq, hd]
    auto q3 = q.view({q_seq,  (int64_t)numHeads,   (int64_t)headDim}).transpose(0, 1);
    auto k3 = k.view({kv_seq, (int64_t)numKvHeads, (int64_t)headDim}).transpose(0, 1);
    auto v3 = v.view({kv_seq, (int64_t)numKvHeads, (int64_t)headDim}).transpose(0, 1);
    auto out3 = at::scaled_dot_product_attention(
                    q3, k3, v3,
                    /*attn_mask=*/std::nullopt,
                    /*dropout_p=*/0.0,
                    /*is_causal=*/isCausal != 0,
                    /*scale=*/std::nullopt,
                    /*enable_gqa=*/numHeads != numKvHeads);
    // [nH, q_seq, hd] -> [q_seq, nH, hd] -> [q_seq, nH * hd]
    auto out2 = out3.transpose(0, 1).contiguous()
                    .view({q_seq, (int64_t)numHeads * (int64_t)headDim});
    return from_tensor(std::move(out2));
}

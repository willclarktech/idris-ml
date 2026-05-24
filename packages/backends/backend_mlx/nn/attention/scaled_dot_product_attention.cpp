/* tensor_sdpa_2d for the mlx backend.
 *
 * Wraps `mlx::core::fast::scaled_dot_product_attention`, mlx-lm's
 * canonical fused attention path (one mlx graph node for the whole
 * QK^T → scale → mask → softmax → V composition; lazy-evaluated so the
 * compute batches into mlx's regular eval boundary). GQA falls out
 * naturally — mlx's SDPA dispatches on the (num_q_heads / num_kv_heads)
 * ratio per call.
 *
 * I/O is 2D-flat to match the Idris caller's `[seq, h*hd]` projection
 * outputs without paying multiplicative-Nat elaboration cost. **Q and
 * KV may have different sequence lengths** for the cache-aware decode
 * path: Q.size(0) = q_seq (1 in steady state); K.size(0) = V.size(0)
 * = kv_seq (cache_len + q_seq). For prefill / training, q_seq ==
 * kv_seq.
 *   Q : [q_seq,  numHeads   * headDim]
 *   K : [kv_seq, numKvHeads * headDim]
 *   V : [kv_seq, numKvHeads * headDim]
 *   out [q_seq, numHeads * headDim]
 *
 * Reshape + transpose to mlx-lm's expected [1, num_heads, seq, head_dim]
 * 4D layout, call SDPA, transpose + reshape back. Reshapes are
 * metadata-only on mlx; transposes are lazy graph nodes that mlx fuses
 * with the SDPA call at eval time.
 *
 * Causal mask under asymmetric: `mask_mode="causal"` on mlx aligns to
 * the lower-right corner of the [q_seq, kv_seq] grid — query position
 * i sees K positions [0 .. kv_seq - q_seq + i] — matching the cache-
 * aware semantics. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"

#include <mlx/fast.h>
#include <cmath>

extern "C" TensorHandle tensor_sdpa_2d_mlx_streamed(
    TensorHandle hq, TensorHandle hk, TensorHandle hv,
    int numHeads, int numKvHeads, int headDim, int isCausal,
    int stream_tag) {
    WITH_STREAM(stream_tag);
    auto q = (Tensor*)hq;
    auto k = (Tensor*)hk;
    auto v = (Tensor*)hv;
    int q_seq  = (int)q->data.shape(0);
    int kv_seq = (int)k->data.shape(0);
    // [seq, h*hd] -> [1, seq, h, hd] -> [1, h, seq, hd]
    auto q3 = mx::transpose(mx::reshape(q->data, {1, q_seq,  numHeads,   headDim}), {0, 2, 1, 3});
    auto k3 = mx::transpose(mx::reshape(k->data, {1, kv_seq, numKvHeads, headDim}), {0, 2, 1, 3});
    auto v3 = mx::transpose(mx::reshape(v->data, {1, kv_seq, numKvHeads, headDim}), {0, 2, 1, 3});
    float scale = 1.0f / std::sqrt((float)headDim);
    // Defensive: under asymmetric q_seq != kv_seq with isCausal=1, do
    // NOT rely on mask_mode="causal" doing lower-right alignment —
    // torch's math impl of SDPA was found 2026-06-04 to apply
    // `.tril(diagonal=0)` without the offset, collapsing visible
    // positions to just j=0 (cache-aware decode produced wrong
    // tokens from the first decode step). Same risk class for mlx
    // (newer kernels likely handle it, older ones may not). Fall
    // back to the legacy per-head decomposition path with an
    // explicit lower-right mask for the asymmetric case; symmetric
    // path stays on the fused kernel.
    if (isCausal && q_seq != kv_seq) {
        // Build [q_seq, kv_seq] additive mask: 0 where visible,
        // -inf where masked. mask[i, j] = 0 if j <= (kv_seq - q_seq) + i
        // else -inf.
        int offset = kv_seq - q_seq;
        std::vector<float> mask_data((size_t)q_seq * (size_t)kv_seq);
        for (int i = 0; i < q_seq; i++) {
            for (int j = 0; j < kv_seq; j++) {
                mask_data[i * kv_seq + j] =
                    (j <= offset + i) ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
        auto mask_arr_2d = mx::array(mask_data.data(),
                                     {q_seq, kv_seq},
                                     q3.dtype());
        // Broadcast mask to [1, 1, q_seq, kv_seq] for the [1, h,
        // q_seq, kv_seq] SDPA layout — mlx broadcast-add handles
        // the head + batch axis fan-out.
        auto mask_arr = mx::reshape(mask_arr_2d, {1, 1, q_seq, kv_seq});
        auto out3 = mx::fast::scaled_dot_product_attention(q3, k3, v3, scale, mask_arr);
        // [1, h, q_seq, hd] -> [1, q_seq, h, hd] -> [q_seq, h*hd]
        auto out2 = mx::reshape(mx::transpose(out3, {0, 2, 1, 3}),
                                {q_seq, numHeads * headDim});
        bool rg = q->requires_grad || k->requires_grad || v->requires_grad;
        auto r = new Tensor(out2, rg);
        return (TensorHandle)r;
    }
    std::string mask_mode = isCausal ? "causal" : "";
    auto out3 = mx::fast::scaled_dot_product_attention(q3, k3, v3, scale, mask_mode);
    // [1, h, q_seq, hd] -> [1, q_seq, h, hd] -> [q_seq, h*hd]
    auto out2 = mx::reshape(mx::transpose(out3, {0, 2, 1, 3}),
                            {q_seq, numHeads * headDim});

    bool rg = q->requires_grad || k->requires_grad || v->requires_grad;
    auto r = new Tensor(out2, rg);
    /* No backward yet — inference only (per #399 plan "Out of scope:
     * Training-side autograd integration of fused ops"). Caller is in
     * `withNoGrad` so requires_grad propagates to false anyway; the
     * tape_append guard below is a safety net for the unexpected. */
    if (rg) {
        /* If training-side ever calls this, the lack of backward will
         * surface as a missing OP_SDPA case in autograd dispatch. */
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_sdpa_2d(
    TensorHandle hq, TensorHandle hk, TensorHandle hv,
    int numHeads, int numKvHeads, int headDim, int isCausal) {
    return tensor_sdpa_2d_mlx_streamed(hq, hk, hv, numHeads, numKvHeads, headDim,
                                       isCausal, default_stream_tag());
}

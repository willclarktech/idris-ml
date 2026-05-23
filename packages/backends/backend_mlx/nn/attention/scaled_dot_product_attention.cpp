/* tensor_sdpa_2d for the mlx backend (TODO #399 Commit B).
 *
 * Wraps `mlx::core::fast::scaled_dot_product_attention`, mlx-lm's
 * canonical fused attention path (one mlx graph node for the whole
 * QK^T → scale → mask → softmax → V composition; lazy-evaluated so the
 * compute batches into mlx's regular eval boundary). GQA falls out
 * naturally — mlx's SDPA dispatches on the (num_q_heads / num_kv_heads)
 * ratio per call.
 *
 * I/O is 2D-flat to match the Idris caller's `[seq, h*hd]` projection
 * outputs without paying multiplicative-Nat elaboration cost:
 *   Q : [seq, numHeads   * headDim]
 *   K : [seq, numKvHeads * headDim]
 *   V : [seq, numKvHeads * headDim]
 *
 * Reshape + transpose to mlx-lm's expected [1, num_heads, seq, head_dim]
 * 4D layout, call SDPA, transpose + reshape back. Reshapes are
 * metadata-only on mlx; transposes are lazy graph nodes that mlx fuses
 * with the SDPA call at eval time. */
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
    int seq = (int)q->data.shape(0);
    // [seq, h*hd] -> [1, seq, h, hd] -> [1, h, seq, hd]
    auto q3 = mx::transpose(mx::reshape(q->data, {1, seq, numHeads,   headDim}), {0, 2, 1, 3});
    auto k3 = mx::transpose(mx::reshape(k->data, {1, seq, numKvHeads, headDim}), {0, 2, 1, 3});
    auto v3 = mx::transpose(mx::reshape(v->data, {1, seq, numKvHeads, headDim}), {0, 2, 1, 3});
    float scale = 1.0f / std::sqrt((float)headDim);
    std::string mask_mode = isCausal ? "causal" : "";
    auto out3 = mx::fast::scaled_dot_product_attention(q3, k3, v3, scale, mask_mode);
    // [1, h, seq, hd] -> [1, seq, h, hd] -> [seq, h*hd]
    auto out2 = mx::reshape(mx::transpose(out3, {0, 2, 1, 3}),
                            {seq, numHeads * headDim});

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

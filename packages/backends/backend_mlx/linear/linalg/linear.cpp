/* tensor_linear (Wx + b) and tensor_linear_2d (X @ W^T + b) for the
 * mlx backend.
 *
 * tensor_linear decomposes into mv + add so the bias dependency lands
 * on the tape — a fused OP_MV(W,x) form dropped bias from the replay
 * graph (broken: tlinear chains where one tlinear's output is the next
 * tlinear's bias had zero gradients on the inner params).
 *
 * tensor_linear_2d records a single OP_LINEAR_2D with LinearReplayMeta
 * carrying the bias pool index (nullable). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_mv_mlx_streamed(TensorHandle hmat, TensorHandle hvec, int stream_tag);
extern "C" TensorHandle tensor_add_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);

extern "C" TensorHandle tensor_linear_mlx_streamed(TensorHandle hW, TensorHandle hx, TensorHandle hbias, int stream_tag) {
    WITH_STREAM(stream_tag);
    TensorHandle mv_h = tensor_mv_mlx_streamed(hW, hx, stream_tag);
    if (!hbias) return mv_h;
    return tensor_add_mlx_streamed(mv_h, hbias, stream_tag);
}

extern "C" TensorHandle tensor_linear(TensorHandle hW, TensorHandle hx, TensorHandle hbias) {
    return tensor_linear_mlx_streamed(hW, hx, hbias, default_stream_tag());
}

extern "C" TensorHandle tensor_linear_2d_mlx_streamed(TensorHandle hW, TensorHandle hX, TensorHandle hbias, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto W = (Tensor*)hW; auto X = (Tensor*)hX; auto bias = (Tensor*)hbias;
    auto WT = mx::transpose(W->data, {1, 0});
    auto result = mx::matmul(X->data, WT);
    if (bias) result = mx::add(result, bias->data);
    bool rg = W->requires_grad || X->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_LINEAR_2D, r, X, W, 0);
        auto meta = new LinearReplayMeta();
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_linear_2d(TensorHandle hW, TensorHandle hX, TensorHandle hbias) {
    return tensor_linear_2d_mlx_streamed(hW, hX, hbias, default_stream_tag());
}

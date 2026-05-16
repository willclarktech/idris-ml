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
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

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

static void mlx_replay_linear_2d(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    /* a = X [B,i], b = W [o,i]. Y = X @ W^T + bias */
                    auto meta = (LinearReplayMeta*)e.meta;
                    auto WT = mx::transpose(b, {1, 0});
                    auto y = mx::matmul(a, WT);
                    if (meta && meta->bias_pool_idx >= 0)
                        y = mx::add(y, pool[meta->bias_pool_idx]);
                    pool[out] = y;
}
MLX_REGISTER_REPLAY(OP_LINEAR_2D, mlx_replay_linear_2d)

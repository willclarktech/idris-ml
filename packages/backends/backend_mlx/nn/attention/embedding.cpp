/* tensor_embedding for the mlx backend.
 *
 * Indices are cast to int32 (mlx's take expects an integer index
 * array). Output is flattened to [n * embedDim] so the FFI consumer
 * sees a 1D buffer — the Idris layer reshapes back. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_embedding_mlx_streamed(TensorHandle hweight, TensorHandle hindices, int n, int embedDim, int stream_tag) {
    WITH_STREAM(stream_tag);
    (void)n;
    auto weight = (Tensor*)hweight;
    auto indices = (Tensor*)hindices;
    auto idx_int = mx::astype(indices->data, mx::int32);
    auto rows = mx::take(weight->data, idx_int, 0);
    auto result = mx::flatten(rows);

    auto r = new Tensor(result, weight->requires_grad);
    if (weight->requires_grad) {
        auto idx_t = new Tensor(idx_int, false);
        tape_append(OP_EMBEDDING, r, weight, idx_t, (double)embedDim);
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    return tensor_embedding_mlx_streamed(hweight, hindices, n, embedDim, default_stream_tag());
}

/* 2D-returning variant: skips the mx::flatten that the legacy flat
 * path applies. mx::take(weight, idx, 0) returns [n, embedDim] directly. */
extern "C" TensorHandle tensor_embedding_2d_mlx_streamed(TensorHandle hweight, TensorHandle hindices, int n, int embedDim, int stream_tag) {
    WITH_STREAM(stream_tag);
    (void)n;
    auto weight = (Tensor*)hweight;
    auto indices = (Tensor*)hindices;
    auto idx_int = mx::astype(indices->data, mx::int32);
    auto rows = mx::take(weight->data, idx_int, 0);

    auto r = new Tensor(rows, weight->requires_grad);
    if (weight->requires_grad) {
        auto idx_t = new Tensor(idx_int, false);
        tape_append(OP_EMBEDDING_2D, r, weight, idx_t, (double)embedDim);
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_embedding_2d(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    return tensor_embedding_2d_mlx_streamed(hweight, hindices, n, embedDim, default_stream_tag());
}

static void mlx_replay_embedding(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    // a = weight, b = indices (int32), scalar_arg = embedDim
                    auto idx_int = mx::astype(b, mx::int32);
                    auto rows = mx::take(a, idx_int, 0);
                    pool[out] = mx::flatten(rows);
}
MLX_REGISTER_REPLAY(OP_EMBEDDING, mlx_replay_embedding)

static void mlx_replay_embedding_2d(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    auto idx_int = mx::astype(b, mx::int32);
    pool[out] = mx::take(a, idx_int, 0);
}
MLX_REGISTER_REPLAY(OP_EMBEDDING_2D, mlx_replay_embedding_2d)

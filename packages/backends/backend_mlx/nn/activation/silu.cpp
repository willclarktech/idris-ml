/* tensor_silu (Swish) for the mlx backend.
 *
 * silu(x) = x * sigmoid(x). mlx has no native silu; this is the same
 * decomposition torch::silu does internally. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_silu_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto result = mx::multiply(t->data, mx::sigmoid(t->data));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_SILU, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_silu(TensorHandle h) {
    return tensor_silu_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_silu(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    pool[out] = mx::multiply(a, mx::sigmoid(a));
}
MLX_REGISTER_REPLAY(OP_SILU, mlx_replay_silu)

/* tensor_clamp_min for the mlx backend. Implemented as mx::maximum
 * against a dtype-matched scalar; backward (via mx::vjp) zeros the
 * gradient at clamped indices. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_clamp_min_mlx_streamed(TensorHandle h, double min_val, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::maximum(t->data, scalar_like(min_val, t->data)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CLAMP_MIN, r, t, nullptr, min_val);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_clamp_min(TensorHandle h, double min_val) {
    return tensor_clamp_min_mlx_streamed(h, min_val, default_stream_tag());
}

static void mlx_replay_clamp_min(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    pool[out] = mx::maximum(a, scalar_like(e.scalar_arg, a));
}
MLX_REGISTER_REPLAY(OP_CLAMP_MIN, mlx_replay_clamp_min)

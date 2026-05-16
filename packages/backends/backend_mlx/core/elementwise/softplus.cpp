/* tensor_softplus for the mlx backend.
 *
 * Numerically stable softplus: max(0, x) + log(1 + exp(-|x|)). The
 * naive log(1 + exp(x)) overflows in float32 for x > ~88 — and the
 * NTM addressing path multiplies softplus(x) by cosine_sim and feeds
 * softmax, so an overflow there silently produces ±inf inputs to
 * softmax and the whole chain becomes NaN at the working point. The
 * stable form is correct for all x: for large positive x it reduces
 * to x, for large negative x it reduces to exp(x) ≈ 0. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_softplus_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto result = mx::add(mx::maximum(t->data, zero_like(t->data)),
                          mx::log(mx::add(one_like(t->data), mx::exp(mx::negative(mx::abs(t->data))))));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTPLUS, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_softplus(TensorHandle h) {
    return tensor_softplus_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_softplus(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    // Smooth form log(1 + exp(a)) — differentiable everywhere
                    // (d/dx = sigmoid(a)). The numerically-stable composite
                    // max(0,a) + log(1+exp(-|a|)) used in the forward kernel
                    // can't be used here: mx::vjp returns subgradient 0 for the
                    // non-differentiable kink at a=0 in mx::maximum, which would
                    // give a wrong d/dx softplus(0) = 0 instead of 0.5.
                    pool[out] = mx::log(mx::add(one_like(a), mx::exp(a)));
}
MLX_REGISTER_REPLAY(OP_SOFTPLUS, mlx_replay_softplus)

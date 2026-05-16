/* tensor_leaky_relu for the mlx backend.
 *
 * leaky_relu(x, alpha) = max(alpha*x, x). The alpha scalar lives on
 * the tape entry's scalar_arg for backward replay. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_leaky_relu_mlx_streamed(TensorHandle h, double alpha, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto alpha_arr = scalar_like(alpha, t->data);
    auto result = mx::maximum(mx::multiply(alpha_arr, t->data), t->data);
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_LEAKY_RELU, r, t, nullptr, alpha);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_leaky_relu(TensorHandle h, double alpha) {
    return tensor_leaky_relu_mlx_streamed(h, alpha, default_stream_tag());
}

static void mlx_replay_leaky_relu(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    auto alpha = scalar_like(e.scalar_arg, a);
                    pool[out] = mx::maximum(mx::multiply(alpha, a), a);
}
MLX_REGISTER_REPLAY(OP_LEAKY_RELU, mlx_replay_leaky_relu)

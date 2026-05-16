/* tensor_cumprod for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_cumprod_mlx_streamed(TensorHandle ht, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)ht;
    auto result = mx::cumprod(t->data, dim);
    auto r = new Tensor(result, t->requires_grad);
    if (r->requires_grad) tape_append(OP_CUMPROD, r, t, NULL, 0.0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    return tensor_cumprod_mlx_streamed(ht, dim, default_stream_tag());
}

static void mlx_replay_cumprod(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    pool[out] = mx::cumprod(a, 0);
}
MLX_REGISTER_REPLAY(OP_CUMPROD, mlx_replay_cumprod)

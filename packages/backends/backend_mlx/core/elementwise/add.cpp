/* tensor_add for the mlx backend. Streamed variant opens the chosen
 * mx::StreamContext, computes the forward via mx::add, allocates a new
 * Tensor (refcount=0; the FFI wrap caller takes the first retain), and
 * appends the op to the tape when either input requires_grad. Backward
 * is replay-based — mx::vjp dispatches on OP_ADD during tape_backward. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_add_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::add(a->data, b->data), rg);
    if (rg) tape_append(OP_ADD, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_add(TensorHandle ha, TensorHandle hb) {
    return tensor_add_mlx_streamed(ha, hb, default_stream_tag());
}

static void mlx_replay_add(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    pool[out] = mx::add(a, b);
}
MLX_REGISTER_REPLAY(OP_ADD, mlx_replay_add)

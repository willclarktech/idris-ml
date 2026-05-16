/* tensor_matmul for the mlx backend. mx::matmul handles arbitrary rank
 * (broadcasts the leading dims) — backward replay reuses OP_MM since
 * both forms are mathematically equivalent for vjp purposes. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_matmul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_matmul(TensorHandle ha, TensorHandle hb) {
    return tensor_matmul_mlx_streamed(ha, hb, default_stream_tag());
}

static void mlx_replay_mm(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    pool[out] = mx::matmul(a, b);
}
MLX_REGISTER_REPLAY(OP_MM, mlx_replay_mm)
MLX_REGISTER_REPLAY(OP_BMM, mlx_replay_mm)
MLX_REGISTER_REPLAY(OP_BMM_3X3, mlx_replay_mm)

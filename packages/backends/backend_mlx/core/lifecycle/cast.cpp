/* tensor_cast_dtype_{f32,f64} for the mlx backend.
 *
 * mx::astype builds a new node in mlx's autograd graph; the
 * OP_CAST_DTYPE tape entry's scalar_arg encodes the target dtype for
 * replay (0.0 = f32, 1.0 = f64). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_cast_dtype_f32_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::astype(t->data, mx::float32), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 0.0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cast_dtype_f32(TensorHandle h) {
    return tensor_cast_dtype_f32_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_cast_dtype_f64_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::astype(t->data, mx::float64), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 1.0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cast_dtype_f64(TensorHandle h) {
    return tensor_cast_dtype_f64_mlx_streamed(h, default_stream_tag());
}

static void mlx_replay_cast_dtype(std::vector<mx::array>& pool, TapeEntry& e) {
    int out = e.result->pool_idx;
    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
    mx::Dtype target = (e.scalar_arg == 0.0 ? mx::float32 : mx::float64);
                    pool[out] = mx::astype(a, target);
}
MLX_REGISTER_REPLAY(OP_CAST_DTYPE, mlx_replay_cast_dtype)

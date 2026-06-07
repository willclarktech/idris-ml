/* tensor_cast_dtype_{f32,f64,bf16,f16,i32} for the mlx backend.
 *
 * mx::astype builds a new node in mlx's autograd graph; the
 * OP_CAST_DTYPE tape entry's scalar_arg encodes the target dtype for
 * replay (0.0 = f32, 1.0 = f64, 2.0 = bf16, 3.0 = f16, 4.0 = i32). */
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

extern "C" TensorHandle tensor_cast_dtype_bf16_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto r = new Tensor(mx::astype(t->data, mx::bfloat16), t->requires_grad);
	if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 2.0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cast_dtype_f16_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto r = new Tensor(mx::astype(t->data, mx::float16), t->requires_grad);
	if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 3.0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cast_dtype_i32_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto r = new Tensor(mx::astype(t->data, mx::int32), t->requires_grad);
	if (t->requires_grad) tape_append(OP_CAST_DTYPE, r, t, nullptr, 4.0);
	return (TensorHandle)r;
}

static void mlx_replay_cast_dtype(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	mx::Dtype target = (e.scalar_arg == 0.0)   ? mx::float32
	                   : (e.scalar_arg == 1.0) ? mx::float64
	                   : (e.scalar_arg == 2.0) ? mx::bfloat16
	                   : (e.scalar_arg == 3.0) ? mx::float16
	                                           : mx::int32; /* 4.0 */
	pool[out] = mx::astype(a, target);
}
MLX_REGISTER_REPLAY(OP_CAST_DTYPE, mlx_replay_cast_dtype)

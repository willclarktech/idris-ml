/* tensor_item for the mlx backend.
 *
 * Forces lazy evaluation before the host readback, then branches on
 * the array's dtype to pick the right scalar type.
 *
 * Until 2026-05-31 this read non-F64 storage via `item<float>()`
 * unconditionally — which silently returned garbage for BF16 storage
 * because mlx's BF16 elements are 2-byte `bfloat16_t`, not 4-byte
 * float. The 16 useful bits got interpreted as the upper half of a
 * 32-bit float (the lower 16 bits were whatever happened to be in
 * the next buffer slot), producing absurd denormal-range values like
 * `2.3e-41` for an actual `1.1` BF16 scalar. The Supervised example
 * on `MLX_DTYPE=BF16` exhibited this as "loss=2.3e-41 from epoch 1"
 * — see the 2026-05-31 commit log. */
#include "../../tensor.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" double tensor_item_mlx_streamed(TensorHandle h, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	mx::eval(t->data);
	if (t->data.dtype() == mx::float64) return t->data.item<double>();
	if (t->data.dtype() == mx::bfloat16) return (double)(float)t->data.item<mx::bfloat16_t>();
	if (t->data.dtype() == mx::float16) return (double)(float)t->data.item<mx::float16_t>();
	if (t->data.dtype() == mx::int32) return (double)t->data.item<int32_t>();
	return (double)t->data.item<float>();
}

extern "C" double tensor_item(TensorHandle h) {
	return tensor_item_mlx_streamed(h, default_stream_tag());
}

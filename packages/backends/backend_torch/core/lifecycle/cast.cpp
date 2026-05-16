/* tensor_cast_dtype_{f32,f64} for the torch backend.
 *
 * at::Tensor::to(dtype) is autograd-traced when both source and target
 * are floating-point types, so gradients flow through the cast
 * naturally. The tracking variant (from_tensor) is used because the
 * cast result is treated as a computation intermediate. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_cast_dtype_f32(TensorHandle src) {
    return from_tensor(to_tensor(src)->to(torch::kFloat32));
}

extern "C" TensorHandle tensor_cast_dtype_f64(TensorHandle src) {
    return from_tensor(to_tensor(src)->to(torch::kFloat64));
}

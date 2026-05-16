/* Tensor accessors — mlx.
 *
 *   - tensor_numel / tensor_dim / tensor_size: shape introspection.
 *   - tensor_to_doubles / tensor_to_floats: host readback bridges.
 *     `tensor_to_floats` fast-paths F32 (memcpy-style loop over the
 *     native buffer); F64 goes through the lingua-franca cast.
 *   - tensor_to_int64: byte-level I64 readout. mlx has no native int64
 *     storage; integer round-trip goes through double, so this inherits
 *     the 2^53 ceiling. Implemented for symbol completeness — the
 *     Idris-side `Compatible MlxDev I64` is closed, so the realistic
 *     reachable use is safetensors I/O on F32/F64-typed tensors.
 *   - tensor_dtype_name: F32 or F64; mlx storage doesn't support other
 *     dtypes (Metal has no bf16/f16/int storage).
 *
 * `tensor_item` (scalar readout) lives in core/lifecycle/item.cpp.
 */
#include "../../tensor.h"
#include "../../precision.h"
#include <cstdlib>
#include <cstdint>

extern "C" int tensor_numel(TensorHandle h) { return (int)((Tensor*)h)->data.size(); }
extern "C" int tensor_dim(TensorHandle h) { return (int)((Tensor*)h)->data.ndim(); }
extern "C" int tensor_size(TensorHandle h, int dim) { return (int)((Tensor*)h)->data.shape(dim); }

extern "C" void tensor_to_doubles(TensorHandle h, double* out) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    mx_to_doubles(t->data, out);
}

// Byte-level I64 readout — declared in backend.h with the byte-exact
// contract honoured only on backends with native int64 storage. mlx
// stores only F32/F64; integer storage round-trips through `double`,
// inheriting the same 2^53 ceiling as the lingua-franca path.
// Practically the safetensors I/O caller only reaches this on tensors
// already typed I64, which mlx can't construct (Compatible MlxDev I64
// is closed). Implemented for symbol completeness.
extern "C" void tensor_to_int64(TensorHandle h, int64_t* out) {
    auto t = (Tensor*)h;
    int n = (int)t->data.size();
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    if (!tmp) return;
    mx::eval(t->data);
    mx_to_doubles(t->data, tmp);
    for (int i = 0; i < n; i++) out[i] = (int64_t)tmp[i];
    free(tmp);
}

extern "C" void tensor_to_floats(TensorHandle h, float* out) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    int n = (int)t->data.size();
    if (t->data.dtype() == mx::float32) {
        const float* src = t->data.data<float>();
        for (int i = 0; i < n; i++) out[i] = src[i];
    } else {
        const double* src = t->data.data<double>();
        for (int i = 0; i < n; i++) out[i] = (float)src[i];
    }
}

extern "C" const char* tensor_dtype_name(TensorHandle h) {
    auto t = (Tensor*)h;
    return (t->data.dtype() == mx::float32) ? "F32" : "F64";
}

/* Tensor accessors — torch.
 *
 *   - tensor_numel / tensor_dim / tensor_size: shape introspection.
 *   - tensor_to_doubles / tensor_to_floats: host readback bridges.
 *     `.cpu()` first — readback to host memory needs the tensor on CPU.
 *     F64 on MPS isn't supported at construction, so the .to(kFloat64)
 *     for an MPS source goes through .cpu() first.
 *   - tensor_to_int64: byte-exact I64 readout — bypasses the double
 *     pivot so values above 2^53 survive. `.to(kInt64)` is a no-op when
 *     the source is already I64; non-I64 sources go through libtorch's
 *     standard truncating cast.
 *   - tensor_dtype_name: maps to_tensor()->scalar_type() to the F32/F64/
 *     BF16/F16/I8/I16/I32/I64/U8/BOOL strings the Idris side expects.
 */
#include "../../tensor.h"
#include <torch/torch.h>
#include <cstring>
#include <cstdint>

extern "C" int tensor_numel(TensorHandle h) {
    return static_cast<int>(to_tensor(h)->numel());
}

extern "C" int tensor_dim(TensorHandle h) {
    return static_cast<int>(to_tensor(h)->dim());
}

extern "C" int tensor_size(TensorHandle h, int dim) {
    return static_cast<int>(to_tensor(h)->size(dim));
}

extern "C" void tensor_to_doubles(TensorHandle h, double* out) {
    // .cpu() before .data_ptr<>() — readback to host memory needs the
    // tensor on CPU. F64 on MPS isn't supported at construction so the
    // .to(kFloat64) for an MPS source goes through .cpu() first.
    auto t = to_tensor(h)->cpu().to(torch::kFloat64).contiguous();
    std::memcpy(out, t.data_ptr<double>(), t.numel() * sizeof(double));
}

extern "C" void tensor_to_floats(TensorHandle h, float* out) {
    auto t = to_tensor(h)->cpu().to(torch::kFloat32).contiguous();
    std::memcpy(out, t.data_ptr<float>(), t.numel() * sizeof(float));
}

// Byte-exact I64 readout — bypasses the double pivot so values
// above 2^53 survive. The `.to(kInt64)` is a no-op when the source
// is already I64; if a caller asks for int64 readout on a non-I64
// tensor we still narrow through torch's standard truncating cast.
// `.cpu()` mirrors the device handling in `tensor_to_doubles`.
extern "C" void tensor_to_int64(TensorHandle h, int64_t* out) {
    auto t = to_tensor(h)->cpu().to(torch::kInt64).contiguous();
    std::memcpy(out, t.data_ptr<int64_t>(), t.numel() * sizeof(int64_t));
}

extern "C" const char* tensor_dtype_name(TensorHandle h) {
    switch (to_tensor(h)->scalar_type()) {
        case torch::kFloat32:  return "F32";
        case torch::kFloat64:  return "F64";
        case torch::kBFloat16: return "BF16";
        case torch::kHalf:     return "F16";
        case torch::kChar:     return "I8";
        case torch::kShort:    return "I16";
        case torch::kInt:      return "I32";
        case torch::kLong:     return "I64";
        case torch::kByte:     return "U8";
        case torch::kBool:     return "BOOL";
        default:               return "F64";
    }
}

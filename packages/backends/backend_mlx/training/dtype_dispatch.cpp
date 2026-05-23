/* Dtype dispatch — mlx.
 *
 * mlx storage natively supports f32, f64 (CPU only — Metal GPU rejects
 * F64 at construction), bfloat16, and float16. The dtag dispatchers
 * route 13→f16, 14→f32, 15→f64, 17→bf16; other dtags (int*, bool) are
 * rejected via `mlx_dtype_unsupported` — symmetric with tape's abort. The Idris `Compatible` gate already prevents
 * unsupported dtags reaching mlx; the abort is a defence-in-depth
 * backstop naming the symbol.
 *
 * Each public `tensor_create_*_streamed` symbol is FFI-visible (extern
 * "C"; consumed by the shared `dtype_streamed.c` TU's port trampolines
 * on backends that opt in — mlx supplies its own implementations
 * directly, which is why these are non-static). The streamed pair pattern
 * is mlx-specific: the trailing `stream_tag` selects the mx::stream the
 * forward op uses; the dtag selects the storage dtype. Base creators
 * (`tensor_create_*_f32_mlx_streamed` / `..._f64_mlx_streamed` /
 * `..._bf16_mlx_streamed`) live alongside in core/lifecycle/ and own
 * the mx::array construction + stream binding + Tensor wrap.
 */
#include "../tensor.h"
#include "../../backend.h"
#include <cstdio>
#include <cstdlib>

/* F32/F64/BF16 base streamed creators — defined in core/lifecycle/. */
extern "C" TensorHandle tensor_create_scalar_f32_mlx_streamed(double value, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_scalar_f64_mlx_streamed(double value, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_scalar_bf16_mlx_streamed(double value, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_scalar_f16_mlx_streamed(double value, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_f32_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_f64_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_bf16_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_f16_mlx_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_1d_f32_mlx_streamed(int n, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_1d_f64_mlx_streamed(int n, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_1d_bf16_mlx_streamed(int n, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_1d_f16_mlx_streamed(int n, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_2d_f32_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_2d_f64_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_2d_bf16_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_2d_f16_mlx_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag);
extern "C" TensorHandle tensor_create_param_1d_f32_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_1d_f64_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_1d_bf16_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_1d_f16_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_2d_f64_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_2d_bf16_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_2d_f16_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_3d_f32_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_3d_f64_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_3d_bf16_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_3d_f16_mlx_streamed(int d0, int d1, int d2, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_4d_f32_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_4d_f64_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_4d_bf16_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_param_4d_f16_mlx_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_1d_f32_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_1d_f64_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_1d_bf16_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_1d_f16_mlx_streamed(int n, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_2d_f32_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_2d_f64_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_2d_bf16_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_create_state_2d_f16_mlx_streamed(int rows, int cols, double* data, int stream_tag);
extern "C" TensorHandle tensor_cast_dtype_f32_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_cast_dtype_f64_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_cast_dtype_bf16_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_cast_dtype_f16_mlx_streamed(TensorHandle h, int stream_tag);

[[noreturn]] static TensorHandle mlx_dtype_unsupported(const char* sym, int dtag) {
    fprintf(stderr,
        "[mlx backend] %s called with dtag=%d. mlx storage supports "
        "f16 (dtag=13), f32 (dtag=14), f64 (dtag=15), and bf16 (dtag=17); "
        "int* + bool are not wired. Bind your code to one of those, or "
        "build with BACKEND=torch for the wider dtype surface.\n", sym, dtag);
    abort();
}

extern "C" TensorHandle tensor_create_scalar_streamed(double value, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_scalar_f32_mlx_streamed(value, requires_grad, stream_tag);
        case 15: return tensor_create_scalar_f64_mlx_streamed(value, requires_grad, stream_tag);
        case 17: return tensor_create_scalar_bf16_mlx_streamed(value, requires_grad, stream_tag);
        case 13: return tensor_create_scalar_f16_mlx_streamed(value, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_scalar_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_f32_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
        case 15: return tensor_create_f64_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
        case 17: return tensor_create_bf16_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
        case 13: return tensor_create_f16_mlx_streamed(data, shape, rank, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_1d_streamed(int n, double* data, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_1d_f32_mlx_streamed(n, data, requires_grad, stream_tag);
        case 15: return tensor_create_1d_f64_mlx_streamed(n, data, requires_grad, stream_tag);
        case 17: return tensor_create_1d_bf16_mlx_streamed(n, data, requires_grad, stream_tag);
        case 13: return tensor_create_1d_f16_mlx_streamed(n, data, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_1d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_2d_f32_mlx_streamed(rows, cols, data, requires_grad, stream_tag);
        case 15: return tensor_create_2d_f64_mlx_streamed(rows, cols, data, requires_grad, stream_tag);
        case 17: return tensor_create_2d_bf16_mlx_streamed(rows, cols, data, requires_grad, stream_tag);
        case 13: return tensor_create_2d_f16_mlx_streamed(rows, cols, data, requires_grad, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_2d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_1d_f32_mlx_streamed(n, data, stream_tag);
        case 15: return tensor_create_param_1d_f64_mlx_streamed(n, data, stream_tag);
        case 17: return tensor_create_param_1d_bf16_mlx_streamed(n, data, stream_tag);
        case 13: return tensor_create_param_1d_f16_mlx_streamed(n, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_1d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_2d_f32_mlx_streamed(rows, cols, data, stream_tag);
        case 15: return tensor_create_param_2d_f64_mlx_streamed(rows, cols, data, stream_tag);
        case 17: return tensor_create_param_2d_bf16_mlx_streamed(rows, cols, data, stream_tag);
        case 13: return tensor_create_param_2d_f16_mlx_streamed(rows, cols, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_2d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_3d_f32_mlx_streamed(d0, d1, d2, data, stream_tag);
        case 15: return tensor_create_param_3d_f64_mlx_streamed(d0, d1, d2, data, stream_tag);
        case 17: return tensor_create_param_3d_bf16_mlx_streamed(d0, d1, d2, data, stream_tag);
        case 13: return tensor_create_param_3d_f16_mlx_streamed(d0, d1, d2, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_3d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_4d_f32_mlx_streamed(d0, d1, d2, d3, data, stream_tag);
        case 15: return tensor_create_param_4d_f64_mlx_streamed(d0, d1, d2, d3, data, stream_tag);
        case 17: return tensor_create_param_4d_bf16_mlx_streamed(d0, d1, d2, d3, data, stream_tag);
        case 13: return tensor_create_param_4d_f16_mlx_streamed(d0, d1, d2, d3, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_param_4d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_1d_f32_mlx_streamed(n, data, stream_tag);
        case 15: return tensor_create_state_1d_f64_mlx_streamed(n, data, stream_tag);
        case 17: return tensor_create_state_1d_bf16_mlx_streamed(n, data, stream_tag);
        case 13: return tensor_create_state_1d_f16_mlx_streamed(n, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_state_1d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_2d_f32_mlx_streamed(rows, cols, data, stream_tag);
        case 15: return tensor_create_state_2d_f64_mlx_streamed(rows, cols, data, stream_tag);
        case 17: return tensor_create_state_2d_bf16_mlx_streamed(rows, cols, data, stream_tag);
        case 13: return tensor_create_state_2d_f16_mlx_streamed(rows, cols, data, stream_tag);
        default: mlx_dtype_unsupported("tensor_create_state_2d_streamed", dtag);
    }
}
extern "C" TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag) {
    switch (dtag) {
        case 14: return tensor_cast_dtype_f32_mlx_streamed(src, stream_tag);
        case 15: return tensor_cast_dtype_f64_mlx_streamed(src, stream_tag);
        case 17: return tensor_cast_dtype_bf16_mlx_streamed(src, stream_tag);
        case 13: return tensor_cast_dtype_f16_mlx_streamed(src, stream_tag);
        default: mlx_dtype_unsupported("tensor_cast_dtype_streamed", dtag);
    }
}

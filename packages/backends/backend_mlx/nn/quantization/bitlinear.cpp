/* nn/quantization/bitlinear.cpp — BitNet b1.58 BitLinear forward (mlx).
 *
 * Storage strategy (see design-decisions.md "Per-backend ternary
 * storage"): on mlx, a Ternary tensor is an `mx::array` with dtype
 * `mx::int8` holding values in {-1, 0, +1}. The 2-bit packed input
 * bytes are unpacked at construction time so the underlying mx::array
 * is framework-native — `mx::matmul` works, autograd-through-dequant-
 * cast works, no parallel mini-tensor wrapper.
 *
 * The 4× memory cost vs tape's packed storage is documented and
 * acceptable for the BitNet-on-24GB test target.
 *
 * Forward: y = (W_ternary.astype(scale.dtype) * scale.unsqueeze(1)) @ x + bias.
 * NoGrad on the weight; the rest of the chain flows through mlx's
 * lazy autograd if scale / x / bias have requires_grad.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_create_ternary_packed_2d_mlx_streamed(
        const uint8_t* packed_bytes, int packed_byte_count,
        int o, int i, int requires_grad, int stream_tag) {
    WITH_STREAM(stream_tag);
    int bytes_per_row = (i + 3) / 4;
    int expected_bytes = bytes_per_row * o;
    if (packed_byte_count != expected_bytes) {
        std::fprintf(stderr, "[mlx] tensor_create_ternary_packed_2d: byte-count "
            "mismatch (got %d, expected %d for shape [%d, %d])\n",
            packed_byte_count, expected_bytes, o, i);
        std::abort();
    }
    if (requires_grad) {
        std::fprintf(stderr, "[mlx] tensor_create_ternary_packed_2d: "
            "requires_grad=1 not supported on int8/ternary storage; "
            "Ternary weights must be NoGrad.\n");
        std::abort();
    }
    std::vector<int8_t> unpacked((size_t)o * (size_t)i);
    for (int j = 0; j < o; j++) {
        const uint8_t* row = packed_bytes + (size_t)j * (size_t)bytes_per_row;
        for (int k = 0; k < i; k++) {
            int byte_idx = k >> 2;
            int slot     = k & 0x3;
            uint8_t code = (uint8_t)((row[byte_idx] >> (slot * 2)) & 0x3u);
            int8_t v;
            switch (code) {
                case 0x0: v =  0; break;
                case 0x1: v =  1; break;
                case 0x3: v = -1; break;
                default:
                    std::fprintf(stderr, "[mlx] tensor_create_ternary_packed_2d: "
                        "invalid 2-bit code 0x%x at row %d col %d\n", code, j, k);
                    std::abort();
            }
            unpacked[(size_t)j * (size_t)i + (size_t)k] = v;
        }
    }
    mx::Shape sh = {o, i};
    auto arr = mx::array(unpacked.data(), sh, mx::int8);
    auto t = new Tensor(arr, /*requires_grad=*/false);
    return (TensorHandle)t;
}

extern "C" TensorHandle tensor_create_ternary_packed_2d(
        const uint8_t* packed_bytes, int packed_byte_count,
        int o, int i, int requires_grad) {
    return tensor_create_ternary_packed_2d_mlx_streamed(
        packed_bytes, packed_byte_count, o, i, requires_grad, default_stream_tag());
}

extern "C" TensorHandle tensor_bitlinear_fwd_mlx_streamed(
        TensorHandle hW, TensorHandle hscale, TensorHandle hx, TensorHandle hbias,
        int stream_tag) {
    WITH_STREAM(stream_tag);
    auto W = (Tensor*)hW;
    auto scale = (Tensor*)hscale;
    auto x = (Tensor*)hx;
    auto bias = hbias ? (Tensor*)hbias : nullptr;
    /* Dequant: int8 → compute dtype, then row-wise scale via broadcast,
       then matmul with x. mx::matmul handles 1D x by treating it as a
       column vector; result is [o]. */
    auto W_dequant = mx::multiply(
        mx::astype(W->data, scale->data.dtype()),
        mx::expand_dims(scale->data, 1));
    auto y = mx::matmul(W_dequant, x->data);
    if (bias) {
        y = mx::add(y, bias->data);
    }
    bool rg = scale->requires_grad || x->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(y, rg);
    /* No tape entry recorded: per-op decomposition above already records
       each sub-op (mx_astype, multiply, expand_dims, matmul, add) on
       mlx's lazy graph. Backward flows through that chain naturally
       for any with-grad inputs. */
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_bitlinear_fwd(
        TensorHandle hW, TensorHandle hscale, TensorHandle hx, TensorHandle hbias) {
    return tensor_bitlinear_fwd_mlx_streamed(hW, hscale, hx, hbias, default_stream_tag());
}

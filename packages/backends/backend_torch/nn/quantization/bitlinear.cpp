/* nn/quantization/bitlinear.cpp — BitNet b1.58 BitLinear forward (torch).
 *
 * Storage strategy (see design-decisions.md "Per-backend ternary
 * storage"): on torch, a Ternary tensor is an `at::Tensor` with
 * `at::ScalarType::Char` (int8) holding values in {-1, 0, +1}. The
 * 2-bit packed input bytes are unpacked at construction time so the
 * underlying at::Tensor is framework-native — `at::matmul` works,
 * autograd-through-dequant-cast works, no parallel mini-tensor
 * wrapper.
 *
 * The 4× memory cost vs tape's packed storage is documented and
 * acceptable for the BitNet-on-24GB test target. Packed storage on
 * torch would require bypassing every framework op + the dispatcher;
 * we don't pay that cost yet.
 *
 * Forward: y = (W_ternary.to(scale.dtype()) * scale.unsqueeze(1)) @ x + bias.
 * NoGrad on the weight (BitNet b1.58 weight is frozen). The cast +
 * mul + matmul still flow through torch's autograd if scale / x have
 * requires_grad — that's intentional, the future training path uses
 * STE on the weight (which sits at NoGrad) and lets gradients
 * accumulate normally on scale / bias / activations.
 */
#include <cstdio>
#include <cstdlib>
#include "../../tensor.h"

extern "C" TensorHandle tensor_create_ternary_packed_2d(
        const uint8_t* packed_bytes, int packed_byte_count,
        int o, int i, int requires_grad) {
    int bytes_per_row = (i + 3) / 4;
    int expected_bytes = bytes_per_row * o;
    if (packed_byte_count != expected_bytes) {
        fprintf(stderr, "[torch] tensor_create_ternary_packed_2d: byte-count "
            "mismatch (got %d, expected %d for shape [%d, %d])\n",
            packed_byte_count, expected_bytes, o, i);
        std::abort();
    }
    auto unpacked = torch::empty({o, i}, at::TensorOptions().dtype(at::kChar));
    int8_t* dst = unpacked.data_ptr<int8_t>();
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
                    fprintf(stderr, "[torch] tensor_create_ternary_packed_2d: "
                        "invalid 2-bit code 0x%x at row %d col %d\n", code, j, k);
                    std::abort();
            }
            dst[(size_t)j * (size_t)i + (size_t)k] = v;
        }
    }
    if (requires_grad) {
        /* int8 dtype rejects requires_grad in torch; flag as a programming
           error rather than silently dropping. Ternary weights are NoGrad
           by construction in the Idris-side type. */
        fprintf(stderr, "[torch] tensor_create_ternary_packed_2d: "
            "requires_grad=1 not supported on int8/ternary storage; "
            "Ternary weights must be NoGrad.\n");
        std::abort();
    }
    return from_tensor_persistent(unpacked);
}

extern "C" TensorHandle tensor_bitlinear_fwd(
        TensorHandle hW, TensorHandle hscale, TensorHandle hx, TensorHandle hbias) {
    auto W = *to_tensor(hW);
    auto scale = *to_tensor(hscale);
    auto x = *to_tensor(hx);
    bool has_bias = hbias != nullptr;
    /* W: [o, i] int8. Dequant via .to(scale.dtype()) — produces a float
       tensor of the same shape, values still in {-1, 0, +1}. Multiply
       by scale unsqueezed to [o, 1] for row-wise broadcast. matmul
       with x [i] -> [o]. */
    auto W_dequant = W.to(scale.scalar_type()) * scale.unsqueeze(1);
    auto y = at::matmul(W_dequant, x);
    if (has_bias) {
        y = y + *to_tensor(hbias);
    }
    return from_tensor(y);
}

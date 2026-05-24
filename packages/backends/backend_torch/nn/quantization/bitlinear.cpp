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
 * Device placement: we unpack on CPU (per-byte loop is fast there),
 * then move the int8 tensor to `g_torch_target_device` so subsequent
 * matmuls against MPS / CUDA activations stay device-consistent.
 * Without this move, a TORCH_DEVICE=mps build crashes inside
 * `addmv_out_mps_impl` with `Expected mat.is_mps() to be true, but
 * got false` because the int8 W stays on CPU while the activations
 * land on MPS.
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

extern c10::Device g_torch_target_device;

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
    if (g_torch_target_device.type() != c10::DeviceType::CPU) {
        unpacked = unpacked.to(g_torch_target_device);
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


/* ------------------------------------------------------------------
   Fused HF BitLinear forward (RMSNorm + act-quant + matmul + bias)
   ------------------------------------------------------------------ */

extern "C" TensorHandle tensor_bitlinear_fwd_hf_quant(
        TensorHandle hW, double w_scale,
        TensorHandle hx, TensorHandle hbias,
        int use_rms_norm, TensorHandle hrms_w, double rms_eps) {
    auto W = *to_tensor(hW);          /* [o, i] int8 */
    auto x = *to_tensor(hx);          /* [i] float */
    bool has_bias = (hbias != nullptr);
    auto dtype = x.scalar_type();
    /* Optional RMSNorm */
    if (use_rms_norm && hrms_w) {
        auto w_rms = *to_tensor(hrms_w);
        auto var = at::mean(x * x);                                       /* scalar */
        auto inv = at::rsqrt(var + rms_eps);
        x = x * inv * w_rms;
    }
    /* Per-token activation quant: scale = 127 / max(|x|, 1e-5). */
    auto xabs_max = at::clamp_min(at::max(at::abs(x)), 1e-5);
    auto in_scale = 127.0 / xabs_max;
    auto x_q = at::clamp(at::round(x * in_scale), -128.0, 127.0);
    /* Matmul + dequant. W.to(dtype) lifts int8 -> float dequant.
       in_scale is a [] (scalar) tensor; we apply w_scale as a multiply
       (matching HF transformers AutoBitLinear.forward, which does
       `output = F.linear(act_quant_dequant(x), w_ternary) * w_scale`).
       Net math: y ≈ (W_ternary @ x) * w_scale (i.e. the effective
       full-precision weight is w_scale * W_ternary). The earlier
       formulation `y_q / (in_scale * w_scale)` divided by w_scale
       instead of multiplying, producing outputs ~w_scale² too small
       per BitLinear and compounding multiplicatively across 30
       decoder blocks. */
    auto y_q = at::matmul(W.to(dtype), x_q);                  /* [o] */
    auto y = y_q * w_scale / in_scale;
    if (has_bias) {
        y = y + *to_tensor(hbias);
    }
    return from_tensor(y);
}


/* ------------------------------------------------------------------
   HF-format ternary load (microsoft/bitnet-b1.58-2B-4T-style checkpoints)
   ------------------------------------------------------------------ */

extern "C" TensorHandle tensor_create_ternary_from_hf_packed_2d(
        const uint8_t* hf_packed_bytes, int o, int i_dim) {
    int hf_row_dim = (o + 3) / 4;
    auto unpacked = torch::empty({o, i_dim}, at::TensorOptions().dtype(at::kChar));
    int8_t* dst = unpacked.data_ptr<int8_t>();
    for (int j = 0; j < o; j++) {
        int hf_chunk = j / hf_row_dim;
        int hf_byte_row = j % hf_row_dim;
        for (int k = 0; k < i_dim; k++) {
            uint8_t hf_byte = hf_packed_bytes[(size_t)hf_byte_row * (size_t)i_dim + (size_t)k];
            int hf_code = (hf_byte >> (2 * hf_chunk)) & 0x3;
            int v = hf_code - 1;
            if (v < -1 || v > 1) {
                std::fprintf(stderr, "[torch] tensor_create_ternary_from_hf_packed_2d: "
                    "invalid HF code %d (byte 0x%02x) at (j=%d, k=%d)\n",
                    hf_code, hf_byte, j, k);
                std::abort();
            }
            dst[(size_t)j * (size_t)i_dim + (size_t)k] = (int8_t)v;
        }
    }
    if (g_torch_target_device.type() != c10::DeviceType::CPU) {
        unpacked = unpacked.to(g_torch_target_device);
    }
    return from_tensor_persistent(unpacked);
}


/* ------------------------------------------------------------------
   Load-time absmean ternary quantization
   ------------------------------------------------------------------ */

extern "C" TensorHandle tensor_absmean_per_row_2d(TensorHandle hw) {
    auto w = *to_tensor(hw);
    if (w.dim() != 2) {
        std::fprintf(stderr, "[torch] tensor_absmean_per_row_2d: expected 2D, "
            "got dim=%lld\n", (long long)w.dim());
        std::abort();
    }
    /* at::mean(abs(w), dim=1) → [o], same dtype as w. NoGrad — one-shot
       frozen-quant calculation. */
    auto scale = at::mean(at::abs(w), /*dim=*/1, /*keepdim=*/false);
    return from_tensor(scale);
}

extern "C" TensorHandle tensor_ternary_quant_with_scale_2d(
        TensorHandle hw, TensorHandle hscale) {
    auto w = *to_tensor(hw);
    auto scale = *to_tensor(hscale);
    if (w.dim() != 2) {
        std::fprintf(stderr, "[torch] tensor_ternary_quant_with_scale_2d: "
            "expected 2D weight, got dim=%lld\n", (long long)w.dim());
        std::abort();
    }
    if (scale.dim() != 1 || scale.size(0) != w.size(0)) {
        std::fprintf(stderr, "[torch] tensor_ternary_quant_with_scale_2d: "
            "scale shape mismatch (expected [%lld], got dim=%lld size0=%lld)\n",
            (long long)w.size(0), (long long)scale.dim(),
            scale.dim() > 0 ? (long long)scale.size(0) : -1);
        std::abort();
    }
    /* Clamp the divisor at 1e-12 — same /0 guard as
       `absmean_ternary_quant` in pytorch/torch_ref/models/bitlinear.py.
       Then mask rows where the *original* scale was zero so they stay
       all-zero post-cast. */
    auto safe = at::clamp_min(scale, 1e-12);
    auto ratio = w / safe.unsqueeze(1);              /* [o, i] */
    auto rounded = at::round(ratio);
    auto clamped = at::clamp(rounded, -1.0, 1.0);
    auto active = (scale > 0).to(w.scalar_type());   /* [o] in w dtype */
    auto t_float = clamped * active.unsqueeze(1);
    auto t_int8 = t_float.to(at::kChar);             /* [o, i] int8 */
    return from_tensor_persistent(t_int8);
}

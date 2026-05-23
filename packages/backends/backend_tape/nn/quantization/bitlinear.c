/* nn/quantization/bitlinear.c — BitNet b1.58 BitLinear forward (inference).
 *
 * Ternary weights stored 2-bits-per-value (4 values per byte, row-major
 * with each row independently packed to `(i + 3) / 4` bytes; trailing
 * bits in the last byte of each row are padded with 00 = ternary 0).
 *
 * Forward: y[j] = scale[j] * sum_i(W_ternary[j, i] * x[i]) + bias[j].
 *
 * Decode happens inline in the inner loop — never materialising a
 * float view of W — which is the whole point of carrying packed
 * ternary on tape (memory + L1 footprint). Real BitNet workloads
 * have i much larger than the L1 line size, so the 2-bit codes
 * dominate the bandwidth even though the multiply is by ±1 or 0.
 *
 * Two compute dtypes are supported: F64 (the tape default) and F32
 * (real `float*` storage via tape's existing F32 arena path —
 * matches the dispatch shape of `tensor_linear` / `tensor_mv`). Both
 * are NoGrad — BitNet b1.58 weight is a frozen quantized param;
 * bias gradient flow lands later if a training path needs it.
 * BF16/F16 inputs are not supported; they would need a separate
 * cast-down step (filed as #411 follow-up if a use case appears).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

/* Build a Ternary tensor from a packed-2-bit byte buffer.
 *
 * Storage: keep the bytes verbatim in the arena (sub-byte storage path —
 * tape pioneers this with #411). Logical shape is [o, i]; numel is the
 * logical element count (o * i), NOT the byte count. The tag
 * `DT_TERNARY` is what tells downstream ops (e.g. `tensor_bitlinear_fwd`
 * below) to decode 2-bit codes rather than reading doubles. */
TensorHandle tensor_create_ternary_packed_2d(
        const uint8_t* packed_bytes, int packed_byte_count,
        int o, int i, int requires_grad) {
    int expected_bytes = ((i + 3) / 4) * o;
    if (packed_byte_count != expected_bytes) {
        fprintf(stderr, "[tape] tensor_create_ternary_packed_2d: byte-count "
            "mismatch (got %d, expected %d for shape [%d, %d])\n",
            packed_byte_count, expected_bytes, o, i);
        abort();
    }
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    uint8_t* data = arena_alloc((size_t)packed_byte_count);
    memcpy(data, packed_bytes, (size_t)packed_byte_count);
    int* shape = arena_alloc(2 * sizeof(int));
    shape[0] = o;
    shape[1] = i;
    t->data = data;
    t->shape = shape;
    t->rank = 2;
    t->numel = o * i;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->grad = NULL;
    t->persistent = 0;
    t->dtype_tag = DT_TERNARY;
    return (TensorHandle)t;
}

/* Decode one ternary slot from row-major packed storage.
 *   row_base: pointer to the start of row j's `bytes_per_row` packed bytes
 *   k: column index within the row (0..i-1)
 * Returns -1, 0, or +1 as an int8_t. */
static inline int8_t decode_slot(const uint8_t* row_base, int k) {
    int byte_idx = k >> 2;        /* k / 4 */
    int slot     = k & 0x3;       /* k % 4 */
    uint8_t code = (uint8_t)((row_base[byte_idx] >> (slot * 2)) & 0x3u);
    switch (code) {
        case 0x0: return  0;
        case 0x1: return  1;
        case 0x3: return -1;
        default:
            fprintf(stderr, "[tape] tensor_bitlinear_fwd: invalid 2-bit code "
                "0x%x at slot %d (byte 0x%02x)\n", code, k, row_base[byte_idx]);
            abort();
    }
}

/* F32 forward — real `float*` data on scale / x / bias / output. Same
   decode-inline inner loop as the F64 path; ternary multiplies are
   exact in both dtypes so the only precision difference is in the
   `scale * sum + bias` accumulate. F32 accumulation can lose a few
   ulps on long inner dims (i >= a few thousand); BitNet workloads
   typically have i ≈ 4k-8k, so an upper-bound max-abs-error in the
   1e-5 / 1e-6 range is expected. */
static TensorHandle tensor_bitlinear_fwd_tape_f32(
        TensorHandle hW, TensorHandle hscale, TensorHandle hx, TensorHandle hbias) {
    Tensor* W = (Tensor*)hW;
    Tensor* scale = (Tensor*)hscale;
    Tensor* x = (Tensor*)hx;
    Tensor* bias = hbias ? (Tensor*)hbias : NULL;
    int o = W->shape[0];
    int i_dim = W->shape[1];
    int bytes_per_row = (i_dim + 3) / 4;
    const uint8_t* W_data = (const uint8_t*)W->data;
    const float* scale_data = (const float*)scale->data;
    const float* x_data = (const float*)x->data;
    const float* bias_data = bias ? (const float*)bias->data : NULL;

    int out_shape[1] = {o};
    float* out_data = arena_alloc((size_t)o * sizeof(float));
    for (int j = 0; j < o; j++) {
        const uint8_t* row = W_data + (size_t)j * (size_t)bytes_per_row;
        float sum = 0.0f;
        for (int k = 0; k < i_dim; k++) {
            int8_t v = decode_slot(row, k);
            if (v != 0) sum += (float)v * x_data[k];
        }
        float y = scale_data[j] * sum;
        if (bias_data) y += bias_data[j];
        out_data[j] = y;
    }
    Tensor* r = make_tensor_arena_f32(out_data, o, out_shape, 1, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_bitlinear_fwd(
        TensorHandle hW, TensorHandle hscale, TensorHandle hx, TensorHandle hbias) {
    Tensor* W = (Tensor*)hW;
    Tensor* scale = (Tensor*)hscale;
    Tensor* x = (Tensor*)hx;
    Tensor* bias = hbias ? (Tensor*)hbias : NULL;

    if (W->dtype_tag != DT_TERNARY) {
        fprintf(stderr, "[tape] tensor_bitlinear_fwd: weight is not Ternary "
            "(dtype_tag=%d). Construct via tensor_create_ternary_packed_2d.\n",
            W->dtype_tag);
        abort();
    }
    /* F32 dispatch: if any of scale/x/bias is F32, require all three. */
    int any_f32 = scale->dtype_tag == DT_F32 || x->dtype_tag == DT_F32 ||
                  (bias && bias->dtype_tag == DT_F32);
    if (any_f32) {
        int all_f32 = scale->dtype_tag == DT_F32 && x->dtype_tag == DT_F32 &&
                      (!bias || bias->dtype_tag == DT_F32);
        if (!all_f32) {
            fprintf(stderr, "[tape] tensor_bitlinear_fwd: mixed-dtype inputs "
                "(scale=%d, x=%d, bias=%d). All of scale/x/bias must share "
                "the same dtype (F32 or F64).\n",
                scale->dtype_tag, x->dtype_tag, bias ? bias->dtype_tag : -1);
            abort();
        }
        return tensor_bitlinear_fwd_tape_f32(hW, hscale, hx, hbias);
    }
    /* F64 path. The lingua-franca on tape means BF16 / F16 inputs would
       arrive here too (their storage is F64 with a narrower dtype_tag);
       the strict equality check rejects them since they need a separate
       cast-down step that this kernel doesn't yet implement. */
    if (scale->dtype_tag != DT_F64 || x->dtype_tag != DT_F64 ||
            (bias && bias->dtype_tag != DT_F64)) {
        fprintf(stderr, "[tape] tensor_bitlinear_fwd: only F64 + F32 inputs "
            "supported (scale=%d, x=%d, bias=%d). BF16 / F16 lands in a "
            "#411 follow-up.\n",
            scale->dtype_tag, x->dtype_tag, bias ? bias->dtype_tag : -1);
        abort();
    }

    int o = W->shape[0];
    int i_dim = W->shape[1];
    int bytes_per_row = (i_dim + 3) / 4;
    const uint8_t* W_data = (const uint8_t*)W->data;
    const double* scale_data = (const double*)scale->data;
    const double* x_data = (const double*)x->data;
    const double* bias_data = bias ? (const double*)bias->data : NULL;

    int out_shape[1] = {o};
    double* out_data = arena_alloc((size_t)o * sizeof(double));
    for (int j = 0; j < o; j++) {
        const uint8_t* row = W_data + (size_t)j * (size_t)bytes_per_row;
        double sum = 0.0;
        for (int k = 0; k < i_dim; k++) {
            int8_t v = decode_slot(row, k);
            if (v != 0) sum += (double)v * x_data[k];
            /* The `v != 0` branch saves ~half the multiplies on real
               BitNet (the trained ternary distribution is roughly
               uniform over {-1, 0, +1}). On a tight loop this is the
               only optimisation that matters; vectorising over the
               packed codes lands in a perf follow-up. */
        }
        double y = scale_data[j] * sum;
        if (bias_data) y += bias_data[j];
        out_data[j] = y;
    }

    /* No tape entry: NoGrad path. BitLinear weight is frozen by
       construction (`Tensor [o, i] d Ternary NoGrad` in Idris), and
       bias-gradient flow is a follow-up if a training path needs it.
       For now the caller's `requires_grad` on scale / x is ignored on
       this op — same shape as the existing F64 inference-only kernels. */
    Tensor* r = make_tensor_arena(out_data, o, out_shape, 1, 0);
    return (TensorHandle)r;
}

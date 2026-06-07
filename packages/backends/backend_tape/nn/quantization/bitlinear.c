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
#include <math.h>
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
TensorHandle tensor_create_ternary_packed_2d(const uint8_t* packed_bytes, int packed_byte_count,
                                             int o, int i, int requires_grad) {
	int expected_bytes = ((i + 3) / 4) * o;
	if (packed_byte_count != expected_bytes) {
		fprintf(stderr,
		        "[tape] tensor_create_ternary_packed_2d: byte-count "
		        "mismatch (got %d, expected %d for shape [%d, %d])\n",
		        packed_byte_count, expected_bytes, o, i);
		// NOLINTNEXTLINE(misc-include-cleaner): macOS SDK: abort via _abort.h umbrella
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
	int byte_idx = k >> 2; /* k / 4 */
	int slot = k & 0x3;    /* k % 4 */
	uint8_t code = (uint8_t)((row_base[byte_idx] >> (slot * 2)) & 0x3u);
	switch (code) {
	case 0x0:
		return 0;
	case 0x1:
		return 1;
	case 0x3:
		return -1;
	default:
		fprintf(stderr,
		        "[tape] tensor_bitlinear_fwd: invalid 2-bit code "
		        "0x%x at slot %d (byte 0x%02x)\n",
		        code, k, row_base[byte_idx]);
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
static TensorHandle tensor_bitlinear_fwd_tape_f32(TensorHandle hW, TensorHandle hscale,
                                                  TensorHandle hx, TensorHandle hbias) {
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

TensorHandle tensor_bitlinear_fwd(TensorHandle hW, TensorHandle hscale, TensorHandle hx,
                                  TensorHandle hbias) {
	Tensor* W = (Tensor*)hW;
	Tensor* scale = (Tensor*)hscale;
	Tensor* x = (Tensor*)hx;
	Tensor* bias = hbias ? (Tensor*)hbias : NULL;

	if (W->dtype_tag != DT_TERNARY) {
		fprintf(stderr,
		        "[tape] tensor_bitlinear_fwd: weight is not Ternary "
		        "(dtype_tag=%d). Construct via tensor_create_ternary_packed_2d.\n",
		        W->dtype_tag);
		abort();
	}
	/* F32 dispatch: if any of scale/x/bias is F32, require all three. */
	int any_f32 =
	    scale->dtype_tag == DT_F32 || x->dtype_tag == DT_F32 || (bias && bias->dtype_tag == DT_F32);
	if (any_f32) {
		int all_f32 = scale->dtype_tag == DT_F32 && x->dtype_tag == DT_F32 &&
		              (!bias || bias->dtype_tag == DT_F32);
		if (!all_f32) {
			fprintf(stderr,
			        "[tape] tensor_bitlinear_fwd: mixed-dtype inputs "
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
		fprintf(stderr,
		        "[tape] tensor_bitlinear_fwd: only F64 + F32 inputs "
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

/* ------------------------------------------------------------------
   Load-time absmean ternary quantization
   ------------------------------------------------------------------ */

/* Per-row absmean: scale[j] = mean_k(|w[j, k]|).
 *
 * `w` must be 2D in F64 or F32. Output matches `w`'s dtype + shape
 * [w->shape[0]]. NoGrad (rhs of the BitNet quantization pipeline). */
TensorHandle tensor_absmean_per_row_2d(TensorHandle hw) {
	Tensor* w = (Tensor*)hw;
	if (w->rank != 2) {
		fprintf(stderr,
		        "[tape] tensor_absmean_per_row_2d: expected 2D input, "
		        "got rank=%d\n",
		        w->rank);
		abort();
	}
	int o = w->shape[0];
	int i_dim = w->shape[1];
	int out_shape[1] = {o};

	if (w->dtype_tag == DT_F64) {
		const double* wd = (const double*)w->data;
		double* sd = arena_alloc((size_t)o * sizeof(double));
		for (int j = 0; j < o; j++) {
			double s = 0.0;
			const double* row = wd + (size_t)j * (size_t)i_dim;
			for (int k = 0; k < i_dim; k++) {
				double v = row[k];
				s += v < 0.0 ? -v : v;
			}
			sd[j] = s / (double)i_dim;
		}
		Tensor* r = make_tensor_arena(sd, o, out_shape, 1, 0);
		return (TensorHandle)r;
	}
	if (w->dtype_tag == DT_F32) {
		const float* wf = (const float*)w->data;
		float* sf = arena_alloc((size_t)o * sizeof(float));
		for (int j = 0; j < o; j++) {
			float s = 0.0f;
			const float* row = wf + (size_t)j * (size_t)i_dim;
			for (int k = 0; k < i_dim; k++) {
				float v = row[k];
				s += v < 0.0f ? -v : v;
			}
			sf[j] = s / (float)i_dim;
		}
		Tensor* r = make_tensor_arena_f32(sf, o, out_shape, 1, 0);
		return (TensorHandle)r;
	}
	{
		fprintf(stderr,
		        "[tape] tensor_absmean_per_row_2d: only F64 and F32 "
		        "inputs supported (got dtype_tag=%d)\n",
		        w->dtype_tag);
		abort();
	}
}

/* HF -> ours layout repack. Reads HF's `[(o+3)/4, i]` uint8 buffer
   (HF encoding: value+1 packed in 2-bit slots, low bits = row 0..row_dim-1,
   etc.) and produces our `[o, (i+3)/4]` packed-ternary tensor with the
   two's-complement codes the rest of the kernels expect. One-shot at
   load time; not a hot path. */
TensorHandle tensor_create_ternary_from_hf_packed_2d(const uint8_t* hf_packed_bytes, int o,
                                                     int i_dim) {
	int hf_row_dim = (o + 3) / 4;
	int our_bytes_per_row = (i_dim + 3) / 4;
	int total_bytes = o * our_bytes_per_row;
	Tensor* t = arena_alloc(sizeof(Tensor));
	memset(t, 0, sizeof(Tensor));
	uint8_t* packed = arena_alloc((size_t)total_bytes);
	memset(packed, 0, (size_t)total_bytes);
	int* shape_out = arena_alloc(2 * sizeof(int));
	shape_out[0] = o;
	shape_out[1] = i_dim;
	t->data = packed;
	t->shape = shape_out;
	t->rank = 2;
	t->numel = o * i_dim;
	t->requires_grad = 0;
	t->tape_idx = -1;
	t->grad = NULL;
	t->persistent = 0;
	t->dtype_tag = DT_TERNARY;
	for (int j = 0; j < o; j++) {
		int hf_chunk = j / hf_row_dim;
		int hf_byte_row = j % hf_row_dim;
		for (int k = 0; k < i_dim; k++) {
			uint8_t hf_byte = hf_packed_bytes[(size_t)hf_byte_row * (size_t)i_dim + (size_t)k];
			int hf_code = (hf_byte >> (2 * hf_chunk)) & 0x3;
			int value = hf_code - 1; /* {-1, 0, +1} */
			if (value < -1 || value > 1) {
				fprintf(stderr,
				        "[tape] tensor_create_ternary_from_hf_packed_2d: "
				        "invalid HF code %d (byte 0x%02x, chunk %d) at (j=%d, k=%d)\n",
				        hf_code, hf_byte, hf_chunk, j, k);
				abort();
			}
			/* Canonical 3-way ternary encoding (HF -1/0/+1 → 11/00/01) */
			// NOLINTNEXTLINE(readability-avoid-nested-conditional-operator)
			uint8_t our_code = (value == 0) ? 0u : (value == 1 ? 1u : 3u);
			int our_byte_idx = j * our_bytes_per_row + (k >> 2);
			int our_slot = k & 0x3;
			packed[our_byte_idx] |= (uint8_t)(our_code << (our_slot * 2));
		}
	}
	return (TensorHandle)t;
}

/* Quantize a 2D float weight to ternary via a pre-computed per-row scale.
 *
 * For each (j, k): t[j, k] = round(w[j, k] / scale[j]).clamp(-1, +1)
 * (rows with scale == 0 produce all-zero ternary).
 *
 * Output: DT_TERNARY in tape's packed 2-bit layout
 * ([o, (i + 3) / 4] bytes). NoGrad. */
static inline int8_t round_clamp_ternary(double x) {
	/* round-half-to-even (banker's) — matches `torch.round` and the
	   absmean_ternary_quant reference in pytorch/torch_ref/models/bitlinear.py. */
	double r = (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
	if (r > 1.0) r = 1.0;
	if (r < -1.0) r = -1.0;
	return (int8_t)r;
}

TensorHandle tensor_ternary_quant_with_scale_2d(TensorHandle hw, TensorHandle hscale) {
	Tensor* w = (Tensor*)hw;
	Tensor* scale = (Tensor*)hscale;
	if (w->rank != 2) {
		fprintf(stderr,
		        "[tape] tensor_ternary_quant_with_scale_2d: expected "
		        "2D weight, got rank=%d\n",
		        w->rank);
		abort();
	}
	if (scale->rank != 1 || scale->shape[0] != w->shape[0]) {
		fprintf(stderr,
		        "[tape] tensor_ternary_quant_with_scale_2d: scale shape "
		        "mismatch (expected [%d], got rank=%d shape0=%d)\n",
		        w->shape[0], scale->rank, scale->rank > 0 ? scale->shape[0] : -1);
		abort();
	}
	if (w->dtype_tag != scale->dtype_tag) {
		fprintf(stderr,
		        "[tape] tensor_ternary_quant_with_scale_2d: dtype "
		        "mismatch (w=%d, scale=%d)\n",
		        w->dtype_tag, scale->dtype_tag);
		abort();
	}
	int o = w->shape[0];
	int i_dim = w->shape[1];
	int bytes_per_row = (i_dim + 3) / 4;
	int total_bytes = bytes_per_row * o;

	/* Output tensor: tape's packed-ternary storage. */
	Tensor* t = arena_alloc(sizeof(Tensor));
	memset(t, 0, sizeof(Tensor));
	uint8_t* packed = arena_alloc((size_t)total_bytes);
	memset(packed, 0, (size_t)total_bytes); /* default code 00 == ternary 0 */
	int* shape_out = arena_alloc(2 * sizeof(int));
	shape_out[0] = o;
	shape_out[1] = i_dim;
	t->data = packed;
	t->shape = shape_out;
	t->rank = 2;
	t->numel = o * i_dim;
	t->requires_grad = 0;
	t->tape_idx = -1;
	t->grad = NULL;
	t->persistent = 0;
	t->dtype_tag = DT_TERNARY;

	if (w->dtype_tag == DT_F64) {
		const double* wd = (const double*)w->data;
		const double* sd = (const double*)scale->data;
		for (int j = 0; j < o; j++) {
			double sj = sd[j];
			uint8_t* row_out = packed + (size_t)j * (size_t)bytes_per_row;
			const double* row_in = wd + (size_t)j * (size_t)i_dim;
			if (sj <= 0.0) continue; /* scale==0 row stays all-zero */
			double inv = 1.0 / sj;
			for (int k = 0; k < i_dim; k++) {
				int8_t v = round_clamp_ternary(row_in[k] * inv);
				/* Encode: 0 -> 00, +1 -> 01, -1 -> 11. Slot 0 in low bits. */
				// NOLINTNEXTLINE(readability-avoid-nested-conditional-operator)
				uint8_t code = (v == 0) ? 0u : (v == 1 ? 1u : 3u);
				int byte_idx = k >> 2;
				int slot = k & 0x3;
				row_out[byte_idx] |= (uint8_t)(code << (slot * 2));
			}
		}
	} else if (w->dtype_tag == DT_F32) {
		const float* wf = (const float*)w->data;
		const float* sf = (const float*)scale->data;
		for (int j = 0; j < o; j++) {
			float sj = sf[j];
			uint8_t* row_out = packed + (size_t)j * (size_t)bytes_per_row;
			const float* row_in = wf + (size_t)j * (size_t)i_dim;
			if (sj <= 0.0f) continue;
			float inv = 1.0f / sj;
			for (int k = 0; k < i_dim; k++) {
				int8_t v = round_clamp_ternary((double)(row_in[k] * inv));
				/* Canonical 3-way ternary encoding (-1/0/+1 → 11/00/01) */
				// NOLINTNEXTLINE(readability-avoid-nested-conditional-operator)
				uint8_t code = (v == 0) ? 0u : (v == 1 ? 1u : 3u);
				int byte_idx = k >> 2;
				int slot = k & 0x3;
				row_out[byte_idx] |= (uint8_t)(code << (slot * 2));
			}
		}
	} else {
		fprintf(stderr,
		        "[tape] tensor_ternary_quant_with_scale_2d: only F64 "
		        "and F32 inputs supported (got dtype_tag=%d)\n",
		        w->dtype_tag);
		abort();
	}
	return (TensorHandle)t;
}

/* ------------------------------------------------------------------
   Fused HF BitLinear forward (RMSNorm + act-quant + matmul + bias)
   ------------------------------------------------------------------ */

/* Tape F64 path. Same decode-inline inner loop as `tensor_bitlinear_fwd`
   but with the activation quantization + RMSNorm + scalar weight_scale
   fused in. The math (per HF transformers' `AutoBitLinear.forward`,
   integrations/bitnet.py:299-312):

     if use_rms_norm: x = x * rsqrt(mean(x^2) + eps) * rms_norm_w
     in_scale = 127 / max(|x|, 1e-5)
     x_q[k] = round(x[k] * in_scale).clamp(-128, 127)
     y[j] = (sum_k W_ternary[j, k] * x_q[k]) * w_scale / in_scale
            + bias[j] if bias else 0

   The `* w_scale / in_scale` factor matches HF's `output * weight_scale`
   on the ActQuant-dequantized input (act_quant_dequant(x) ≈ x_q /
   in_scale). The earlier `/ (in_scale * w_scale)` formulation divided
   by w_scale instead of multiplying — wrong by w_scale² per BitLinear.

   Output y is F64. */
static TensorHandle tensor_bitlinear_fwd_hf_quant_tape_f64(TensorHandle hW, double w_scale,
                                                           TensorHandle hx, TensorHandle hbias,
                                                           int use_rms_norm, TensorHandle hrms_w,
                                                           double rms_eps) {
	Tensor* W = (Tensor*)hW;
	Tensor* x = (Tensor*)hx;
	Tensor* bias = hbias ? (Tensor*)hbias : NULL;
	Tensor* rms_w = (use_rms_norm && hrms_w) ? (Tensor*)hrms_w : NULL;
	int o = W->shape[0];
	int i_dim = W->shape[1];
	int bytes_per_row = (i_dim + 3) / 4;
	const uint8_t* W_data = (const uint8_t*)W->data;
	const double* x_data = (const double*)x->data;
	const double* bias_data = bias ? (const double*)bias->data : NULL;
	const double* rms_w_data = rms_w ? (const double*)rms_w->data : NULL;

	/* Optional RMSNorm */
	double* xn = malloc((size_t)i_dim * sizeof(double));
	if (use_rms_norm && rms_w_data) {
		double ss = 0.0;
		for (int k = 0; k < i_dim; k++)
			ss += x_data[k] * x_data[k];
		double inv = 1.0 / sqrt(ss / (double)i_dim + rms_eps);
		for (int k = 0; k < i_dim; k++)
			xn[k] = x_data[k] * inv * rms_w_data[k];
	} else {
		for (int k = 0; k < i_dim; k++)
			xn[k] = x_data[k];
	}

	/* Activation quant */
	double xmax = 0.0;
	for (int k = 0; k < i_dim; k++) {
		double a = xn[k] < 0 ? -xn[k] : xn[k];
		if (a > xmax) xmax = a;
	}
	if (xmax < 1.0e-5) xmax = 1.0e-5;
	double in_scale = 127.0 / xmax;
	double* xq = malloc((size_t)i_dim * sizeof(double));
	for (int k = 0; k < i_dim; k++) {
		double v = rint(xn[k] * in_scale);
		if (v < -128.0) v = -128.0;
		if (v > 127.0) v = 127.0;
		xq[k] = v;
	}
	free(xn);

	/* Matmul with ternary W (decode inline), then dequant + bias.
	   Apply `* w_scale / in_scale` (matches HF AutoBitLinear). */
	double rescale = w_scale / in_scale;
	int out_shape[1] = {o};
	double* out_data = arena_alloc((size_t)o * sizeof(double));
	for (int j = 0; j < o; j++) {
		const uint8_t* row = W_data + (size_t)j * (size_t)bytes_per_row;
		double sum = 0.0;
		for (int k = 0; k < i_dim; k++) {
			int8_t v = decode_slot(row, k);
			if (v != 0) sum += (double)v * xq[k];
		}
		double y = sum * rescale;
		if (bias_data) y += bias_data[j];
		out_data[j] = y;
	}
	free(xq);
	Tensor* r = make_tensor_arena(out_data, o, out_shape, 1, 0);
	return (TensorHandle)r;
}

/* Tape F32 path — same loop with float storage. */
static TensorHandle tensor_bitlinear_fwd_hf_quant_tape_f32(TensorHandle hW, double w_scale,
                                                           TensorHandle hx, TensorHandle hbias,
                                                           int use_rms_norm, TensorHandle hrms_w,
                                                           double rms_eps) {
	Tensor* W = (Tensor*)hW;
	Tensor* x = (Tensor*)hx;
	Tensor* bias = hbias ? (Tensor*)hbias : NULL;
	Tensor* rms_w = (use_rms_norm && hrms_w) ? (Tensor*)hrms_w : NULL;
	int o = W->shape[0];
	int i_dim = W->shape[1];
	int bytes_per_row = (i_dim + 3) / 4;
	const uint8_t* W_data = (const uint8_t*)W->data;
	const float* x_data = (const float*)x->data;
	const float* bias_data = bias ? (const float*)bias->data : NULL;
	const float* rms_w_data = rms_w ? (const float*)rms_w->data : NULL;

	float* xn = malloc((size_t)i_dim * sizeof(float));
	if (use_rms_norm && rms_w_data) {
		float ss = 0.0f;
		for (int k = 0; k < i_dim; k++)
			ss += x_data[k] * x_data[k];
		float inv = 1.0f / sqrtf(ss / (float)i_dim + (float)rms_eps);
		for (int k = 0; k < i_dim; k++)
			xn[k] = x_data[k] * inv * rms_w_data[k];
	} else {
		for (int k = 0; k < i_dim; k++)
			xn[k] = x_data[k];
	}
	float xmax = 0.0f;
	for (int k = 0; k < i_dim; k++) {
		float a = xn[k] < 0 ? -xn[k] : xn[k];
		if (a > xmax) xmax = a;
	}
	if (xmax < 1.0e-5f) xmax = 1.0e-5f;
	float in_scale_f = 127.0f / xmax;
	float* xq = malloc((size_t)i_dim * sizeof(float));
	for (int k = 0; k < i_dim; k++) {
		float v = rintf(xn[k] * in_scale_f);
		if (v < -128.0f) v = -128.0f;
		if (v > 127.0f) v = 127.0f;
		xq[k] = v;
	}
	free(xn);
	float rescale = (float)w_scale / in_scale_f;
	int out_shape[1] = {o};
	float* out_data = arena_alloc((size_t)o * sizeof(float));
	for (int j = 0; j < o; j++) {
		const uint8_t* row = W_data + (size_t)j * (size_t)bytes_per_row;
		float sum = 0.0f;
		for (int k = 0; k < i_dim; k++) {
			int8_t v = decode_slot(row, k);
			if (v != 0) sum += (float)v * xq[k];
		}
		float y = sum * rescale;
		if (bias_data) y += bias_data[j];
		out_data[j] = y;
	}
	free(xq);
	Tensor* r = make_tensor_arena_f32(out_data, o, out_shape, 1, 0);
	return (TensorHandle)r;
}

TensorHandle tensor_bitlinear_fwd_hf_quant(TensorHandle hW, double w_scale, TensorHandle hx,
                                           TensorHandle hbias, int use_rms_norm,
                                           TensorHandle hrms_w, double rms_eps) {
	Tensor* x = (Tensor*)hx;
	if (x->dtype_tag == DT_F32) {
		return tensor_bitlinear_fwd_hf_quant_tape_f32(hW, w_scale, hx, hbias, use_rms_norm, hrms_w,
		                                              rms_eps);
	}
	if (x->dtype_tag != DT_F64) {
		fprintf(stderr,
		        "[tape] tensor_bitlinear_fwd_hf_quant: only F64 + F32 "
		        "supported (x dtype_tag=%d)\n",
		        x->dtype_tag);
		abort();
	}
	return tensor_bitlinear_fwd_hf_quant_tape_f64(hW, w_scale, hx, hbias, use_rms_norm, hrms_w,
	                                              rms_eps);
}

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

extern "C" TensorHandle tensor_create_ternary_packed_2d_mlx_streamed(const uint8_t* packed_bytes,
                                                                     int packed_byte_count, int o,
                                                                     int i, int requires_grad,
                                                                     int stream_tag) {
	WITH_STREAM(stream_tag);
	int bytes_per_row = (i + 3) / 4;
	int expected_bytes = bytes_per_row * o;
	if (packed_byte_count != expected_bytes) {
		std::fprintf(stderr,
		             "[mlx] tensor_create_ternary_packed_2d: byte-count "
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
			int slot = k & 0x3;
			uint8_t code = (uint8_t)((row[byte_idx] >> (slot * 2)) & 0x3u);
			int8_t v;
			switch (code) {
			case 0x0:
				v = 0;
				break;
			case 0x1:
				v = 1;
				break;
			case 0x3:
				v = -1;
				break;
			default:
				std::fprintf(stderr,
				             "[mlx] tensor_create_ternary_packed_2d: "
				             "invalid 2-bit code 0x%x at row %d col %d\n",
				             code, j, k);
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

extern "C" TensorHandle tensor_create_ternary_packed_2d(const uint8_t* packed_bytes,
                                                        int packed_byte_count, int o, int i,
                                                        int requires_grad) {
	return tensor_create_ternary_packed_2d_mlx_streamed(packed_bytes, packed_byte_count, o, i,
	                                                    requires_grad, default_stream_tag());
}

extern "C" TensorHandle tensor_bitlinear_fwd_mlx_streamed(TensorHandle hW, TensorHandle hscale,
                                                          TensorHandle hx, TensorHandle hbias,
                                                          int stream_tag) {
	WITH_STREAM(stream_tag);
	auto W = (Tensor*)hW;
	auto scale = (Tensor*)hscale;
	auto x = (Tensor*)hx;
	auto bias = hbias ? (Tensor*)hbias : nullptr;
	/* Dequant: int8 → compute dtype, then row-wise scale via broadcast,
	   then matmul with x. mx::matmul handles 1D x by treating it as a
	   column vector; result is [o]. */
	auto W_dequant =
	    mx::multiply(mx::astype(W->data, scale->data.dtype()), mx::expand_dims(scale->data, 1));
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

extern "C" TensorHandle tensor_bitlinear_fwd(TensorHandle hW, TensorHandle hscale, TensorHandle hx,
                                             TensorHandle hbias) {
	return tensor_bitlinear_fwd_mlx_streamed(hW, hscale, hx, hbias, default_stream_tag());
}

/* ------------------------------------------------------------------
   Fused HF BitLinear forward (RMSNorm + act-quant + matmul + bias)
   ------------------------------------------------------------------ */

extern "C" TensorHandle
tensor_bitlinear_fwd_hf_quant_mlx_streamed(TensorHandle hW, double w_scale, TensorHandle hx,
                                           TensorHandle hbias, int use_rms_norm,
                                           TensorHandle hrms_w, double rms_eps, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto W = (Tensor*)hW;
	auto x_t = (Tensor*)hx;
	auto bias = hbias ? (Tensor*)hbias : nullptr;
	auto dtype = x_t->data.dtype();
	auto x = x_t->data;
	/* Optional RMSNorm */
	if (use_rms_norm && hrms_w) {
		auto rms_w = ((Tensor*)hrms_w)->data;
		auto var = mx::mean(mx::multiply(x, x));
		auto eps = mx::astype(mx::array((float)rms_eps), dtype);
		auto inv = mx::rsqrt(mx::add(var, eps));
		x = mx::multiply(mx::multiply(x, inv), rms_w);
	}
	/* Per-token activation quant */
	auto xabs_max = mx::maximum(mx::max(mx::abs(x)), mx::astype(mx::array(1.0e-5f), dtype));
	auto in_scale = mx::divide(mx::astype(mx::array(127.0f), dtype), xabs_max);
	auto x_q = mx::clip(mx::round(mx::multiply(x, in_scale)), mx::astype(mx::array(-128.0f), dtype),
	                    mx::astype(mx::array(127.0f), dtype));
	/* Matmul + dequant. Apply w_scale as a MULTIPLY (matching HF
	   transformers' AutoBitLinear.forward: `F.linear(act_quant_dequant(x),
	   w_ternary) * w_scale`). The earlier divide-by-(in_scale * w_scale)
	   form effectively divided by w_scale instead of multiplying,
	   producing outputs ~w_scale² too small per BitLinear. */
	auto W_dequant = mx::astype(W->data, dtype);
	auto y_q = mx::matmul(W_dequant, x_q); /* [o] */
	auto w_scale_t = mx::astype(mx::array((float)w_scale), dtype);
	auto y = mx::divide(mx::multiply(y_q, w_scale_t), in_scale);
	if (bias) {
		y = mx::add(y, bias->data);
	}
	auto r = new Tensor(y, /*requires_grad=*/false);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_bitlinear_fwd_hf_quant(TensorHandle hW, double w_scale,
                                                      TensorHandle hx, TensorHandle hbias,
                                                      int use_rms_norm, TensorHandle hrms_w,
                                                      double rms_eps) {
	return tensor_bitlinear_fwd_hf_quant_mlx_streamed(hW, w_scale, hx, hbias, use_rms_norm, hrms_w,
	                                                  rms_eps, default_stream_tag());
}

/* ------------------------------------------------------------------
   HF-format ternary load (microsoft/bitnet-b1.58-2B-4T-style checkpoints)
   ------------------------------------------------------------------ */

/* HF -> ours: read HF's `[(o+3)/4, i]` uint8 buffer (axis-0 packing,
   value+1 encoding) and produce an int8 [o, i] mx::array. Same int8
   storage as `tensor_create_ternary_packed_2d` on mlx. */
extern "C" TensorHandle
tensor_create_ternary_from_hf_packed_2d_mlx_streamed(const uint8_t* hf_packed_bytes, int o,
                                                     int i_dim, int stream_tag) {
	WITH_STREAM(stream_tag);
	int hf_row_dim = (o + 3) / 4;
	std::vector<int8_t> unpacked((size_t)o * (size_t)i_dim);
	for (int j = 0; j < o; j++) {
		int hf_chunk = j / hf_row_dim;
		int hf_byte_row = j % hf_row_dim;
		for (int k = 0; k < i_dim; k++) {
			uint8_t hf_byte = hf_packed_bytes[(size_t)hf_byte_row * (size_t)i_dim + (size_t)k];
			int hf_code = (hf_byte >> (2 * hf_chunk)) & 0x3;
			int v = hf_code - 1;
			if (v < -1 || v > 1) {
				std::fprintf(stderr,
				             "[mlx] tensor_create_ternary_from_hf_packed_2d: "
				             "invalid HF code %d (byte 0x%02x) at (j=%d, k=%d)\n",
				             hf_code, hf_byte, j, k);
				std::abort();
			}
			unpacked[(size_t)j * (size_t)i_dim + (size_t)k] = (int8_t)v;
		}
	}
	mx::Shape sh = {o, i_dim};
	auto arr = mx::array(unpacked.data(), sh, mx::int8);
	auto t = new Tensor(arr, /*requires_grad=*/false);
	return (TensorHandle)t;
}

extern "C" TensorHandle tensor_create_ternary_from_hf_packed_2d(const uint8_t* hf_packed_bytes,
                                                                int o, int i_dim) {
	return tensor_create_ternary_from_hf_packed_2d_mlx_streamed(hf_packed_bytes, o, i_dim,
	                                                            default_stream_tag());
}

/* ------------------------------------------------------------------
   Load-time absmean ternary quantization
   ------------------------------------------------------------------ */

extern "C" TensorHandle tensor_absmean_per_row_2d_mlx_streamed(TensorHandle hw, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto w = (Tensor*)hw;
	if (w->data.ndim() != 2) {
		std::fprintf(stderr,
		             "[mlx] tensor_absmean_per_row_2d: expected 2D, "
		             "got ndim=%d\n",
		             (int)w->data.ndim());
		std::abort();
	}
	/* mean(abs(w), axis=1, keepdims=false) → [o] in w's dtype.
	   NoGrad — this is a one-shot frozen-quant computation. */
	auto scale = mx::mean(mx::abs(w->data), 1, /*keepdims=*/false);
	auto t = new Tensor(scale, /*requires_grad=*/false);
	return (TensorHandle)t;
}

extern "C" TensorHandle tensor_absmean_per_row_2d(TensorHandle hw) {
	return tensor_absmean_per_row_2d_mlx_streamed(hw, default_stream_tag());
}

extern "C" TensorHandle tensor_ternary_quant_with_scale_2d_mlx_streamed(TensorHandle hw,
                                                                        TensorHandle hscale,
                                                                        int stream_tag) {
	WITH_STREAM(stream_tag);
	auto w = (Tensor*)hw;
	auto scale = (Tensor*)hscale;
	if (w->data.ndim() != 2) {
		std::fprintf(stderr,
		             "[mlx] tensor_ternary_quant_with_scale_2d: expected "
		             "2D weight, got ndim=%d\n",
		             (int)w->data.ndim());
		std::abort();
	}
	if (scale->data.ndim() != 1 || scale->data.shape(0) != w->data.shape(0)) {
		std::fprintf(stderr,
		             "[mlx] tensor_ternary_quant_with_scale_2d: scale "
		             "shape mismatch (expected [%d], got ndim=%d shape0=%d)\n",
		             (int)w->data.shape(0), (int)scale->data.ndim(),
		             scale->data.ndim() > 0 ? (int)scale->data.shape(0) : -1);
		std::abort();
	}
	/* Per-row divisor; guard against /0 by clamping the divisor at a
	   tiny floor (mlx's astype handles inf gracefully but we want the
	   same {-1, 0, +1} clamp behaviour as the tape kernel). The clamp
	   at 1e-12 matches `absmean_ternary_quant`'s `clamp(min=1e-12)` in
	   `pytorch/torch_ref/models/bitlinear.py`. */
	auto safe = mx::maximum(scale->data, mx::astype(mx::array(1e-12f), scale->data.dtype()));
	auto divisor = mx::expand_dims(safe, 1);   /* [o, 1] */
	auto ratio = mx::divide(w->data, divisor); /* [o, i] */
	auto rounded = mx::round(ratio);
	auto clamped = mx::clip(rounded, mx::astype(mx::array(-1.0f), w->data.dtype()),
	                        mx::astype(mx::array(1.0f), w->data.dtype()));
	/* Zero out rows where original scale <= 0 (all-zero rows). */
	auto active = mx::greater(scale->data, mx::astype(mx::array(0.0f), scale->data.dtype()));
	auto mask = mx::expand_dims(mx::astype(active, w->data.dtype()), 1);
	auto t_float = mx::multiply(clamped, mask);
	auto t_int8 = mx::astype(t_float, mx::int8); /* [o, i] int8 */
	auto out = new Tensor(t_int8, /*requires_grad=*/false);
	return (TensorHandle)out;
}

extern "C" TensorHandle tensor_ternary_quant_with_scale_2d(TensorHandle hw, TensorHandle hscale) {
	return tensor_ternary_quant_with_scale_2d_mlx_streamed(hw, hscale, default_stream_tag());
}

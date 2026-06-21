/* torch-only Criterion suite for BitNet b1.58 BitLinear (torch).
 *
 * Targets the value paths the common tape suite never reaches in
 * quantization/bitlinear.cpp (65%):
 *   - tensor_create_ternary_packed_2d 2-bit unpack (codes 0/+1/-1).
 *   - tensor_bitlinear_fwd  : y = (W_ternary.to(scale) * scale) @ x + bias.
 *   - tensor_bitlinear_fwd_hf_quant : RMSNorm + per-token act-quant +
 *     matmul + w_scale dequant + bias (lines 170-* HF path body).
 *   - tensor_create_ternary_from_hf_packed_2d : HF chunk-packed layout.
 *   - tensor_absmean_per_row_2d : per-row mean(|W|) (the torch impl is
 *     real libtorch at::mean — distinct from the tape impl that crashes
 *     on valid 2D input per TODO.md; verified torch-side here).
 *   - tensor_ternary_quant_with_scale_2d : round/clamp/zero-row-mask,
 *     verified via a subsequent forward.
 *
 * torch CPU base dtype is F64; W stays int8 (ternary) and is NoGrad by
 * construction, so these are pure forward value checks at exact / relaxed
 * tolerance. The own-line abort guards (byte-count mismatch, invalid
 * 2-bit code, requires_grad on int8, non-2D input, scale-shape mismatch,
 * MPS device move) are error / MPS-only paths — left to the EXCL markers
 * and uncertainties, not exercised here (libtorch SIGABRT death tests are
 * fragile per the suite policy).
 */

#include <criterion/criterion.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

static double* hcopy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

/* 2-bit pack one row of i values (each in {-1,0,+1}) into bytes.
   code: 0 -> 0x0, +1 -> 0x1, -1 -> 0x3; 4 values per byte, slot*2 shift. */
static void pack_row(const int* vals, int i, uint8_t* dst) {
	int bytes_per_row = (i + 3) / 4;
	for (int b = 0; b < bytes_per_row; b++)
		dst[b] = 0;
	for (int k = 0; k < i; k++) {
		uint8_t code = vals[k] == 1 ? 0x1u : (vals[k] == -1 ? 0x3u : 0x0u);
		dst[k >> 2] |= (uint8_t)(code << ((k & 0x3) * 2));
	}
}

/* tensor_create_ternary_packed_2d + tensor_bitlinear_fwd value check. */
Test(torch_nn_quantization_bitlinear, packed_unpack_then_forward) {
	/* W [2,4]: row0 = [+1,-1,+1, 0], row1 = [0,+1,-1,+1]. */
	int r0[] = {1, -1, 1, 0};
	int r1[] = {0, 1, -1, 1};
	uint8_t packed[2]; /* bytes_per_row = (4+3)/4 = 1 -> 2 bytes total. */
	pack_row(r0, 4, &packed[0]);
	pack_row(r1, 4, &packed[1]);
	TensorHandle W = tensor_create_ternary_packed_2d(packed, /*packed_byte_count=*/2, /*o=*/2,
	                                                 /*i=*/4, /*requires_grad=*/0);
	/* scale [2] = {2.0, 0.5}. */
	double sd[] = {2.0, 0.5};
	TensorHandle scale = tensor_create_1d_f32(2, hcopy(sd, 2), /*rg=*/0);
	/* x [4] = {1, 2, 3, 4}. */
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_1d_f32(4, hcopy(xd, 4), /*rg=*/0);
	/* y0 = 2.0 * ( 1*1 + (-1)*2 + 1*3 + 0*4 ) = 2.0 * 2 = 4.0
	   y1 = 0.5 * ( 0*1 + 1*2 + (-1)*3 + 1*4 ) = 0.5 * 3 = 1.5 */
	TensorHandle y = tensor_bitlinear_fwd(W, scale, x, /*bias=*/NULL);
	cr_assert_eq(tensor_numel(y), 2, "y numel should be 2");
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], 4.0, 1e-5, "y0 exp 4.0 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 1.5, 1e-5, "y1 exp 1.5 got %.6f", buf[1]);
}

/* forward with bias -> the has_bias branch (y = y + bias). */
Test(torch_nn_quantization_bitlinear, forward_with_bias) {
	int r0[] = {1, 0};
	int r1[] = {0, 1};
	uint8_t packed[2]; /* i=2 -> bytes_per_row=1, 2 rows. */
	pack_row(r0, 2, &packed[0]);
	pack_row(r1, 2, &packed[1]);
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 2, /*o=*/2, /*i=*/2, /*rg=*/0);
	double sd[] = {1.0, 1.0};
	TensorHandle scale = tensor_create_1d_f32(2, hcopy(sd, 2), 0);
	double xd[] = {5.0, 7.0};
	TensorHandle x = tensor_create_1d_f32(2, hcopy(xd, 2), 0);
	double bd[] = {10.0, 20.0};
	TensorHandle bias = tensor_create_1d_f32(2, hcopy(bd, 2), 0);
	/* y0 = 1*5 + 10 = 15; y1 = 1*7 + 20 = 27. */
	TensorHandle y = tensor_bitlinear_fwd(W, scale, x, bias);
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], 15.0, 1e-5, "y0 exp 15 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 27.0, 1e-5, "y1 exp 27 got %.6f", buf[1]);
}

/* tensor_bitlinear_fwd_hf_quant without RMSNorm (use_rms_norm=0) — the
   per-token act-quant + matmul + w_scale-dequant + no-bias path. */
Test(torch_nn_quantization_bitlinear, hf_quant_no_rmsnorm) {
	int r0[] = {1, 1};
	int r1[] = {1, -1};
	uint8_t packed[2];
	pack_row(r0, 2, &packed[0]);
	pack_row(r1, 2, &packed[1]);
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 2, /*o=*/2, /*i=*/2, /*rg=*/0);
	double xd[] = {4.0, 2.0};
	TensorHandle x = tensor_create_1d_f32(2, hcopy(xd, 2), 0);
	/* Per-token quant: in_scale = 127 / max(|x|) = 127/4 = 31.75.
	   x_q = round(x * in_scale) = round({127, 63.5}) = {127, 64}.
	   y_q = W @ x_q : row0 = 127+64 = 191; row1 = 127-64 = 63.
	   y = y_q * w_scale / in_scale, with w_scale = 1.0:
	   y0 = 191 / 31.75 ~= 6.0157; y1 = 63 / 31.75 ~= 1.9843. */
	TensorHandle y = tensor_bitlinear_fwd_hf_quant(W, /*w_scale=*/1.0, x, /*bias=*/NULL,
	                                               /*use_rms_norm=*/0, /*rms_w=*/NULL,
	                                               /*rms_eps=*/1e-5);
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], 191.0 / 31.75, 1e-4, "hf y0 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 63.0 / 31.75, 1e-4, "hf y1 got %.6f", buf[1]);
}

/* HF-quant WITH RMSNorm + bias -> the use_rms_norm!=0 branch (lines ~124-128)
   plus the has_bias branch. We assert the RMSNorm normalizes to unit RMS
   when rms_w is all-ones: var = mean(x^2), x_norm = x / sqrt(var). */
Test(torch_nn_quantization_bitlinear, hf_quant_with_rmsnorm_and_bias) {
	int r0[] = {1, 0};
	int r1[] = {0, 1};
	uint8_t packed[2];
	pack_row(r0, 2, &packed[0]);
	pack_row(r1, 2, &packed[1]);
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 2, /*o=*/2, /*i=*/2, /*rg=*/0);
	/* x = {3, 4}: mean(x^2) = (9+16)/2 = 12.5; rms = sqrt(12.5) ~= 3.5355.
	   x_norm ~= {0.84853, 1.13137} (rms_w = ones, eps negligible). */
	double xd[] = {3.0, 4.0};
	TensorHandle x = tensor_create_1d_f32(2, hcopy(xd, 2), 0);
	double rw[] = {1.0, 1.0};
	TensorHandle rms_w = tensor_create_1d_f32(2, hcopy(rw, 2), 0);
	double bd[] = {100.0, 200.0};
	TensorHandle bias = tensor_create_1d_f32(2, hcopy(bd, 2), 0);
	/* After normalize, per-token quant + identity-ish W. W row0 picks
	   x_norm[0], row1 picks x_norm[1]. With w_scale=1 the matmul/dequant
	   round-trips back to ~x_norm. So y ~= x_norm + bias. */
	TensorHandle y = tensor_bitlinear_fwd_hf_quant(W, /*w_scale=*/1.0, x, bias,
	                                               /*use_rms_norm=*/1, rms_w, /*rms_eps=*/1e-5);
	double buf[2];
	tensor_to_doubles(y, buf);
	/* The quant round-trip is lossy (round to int8), so allow a coarse tol;
	   the dominant term is the +100 / +200 bias making this a clean check
	   that the RMSNorm + bias branches both executed and produced a finite
	   value near x_norm + bias. */
	cr_assert_float_eq(buf[0], 0.84853 + 100.0, 1e-1, "rmsnorm y0 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 1.13137 + 200.0, 1e-1, "rmsnorm y1 got %.6f", buf[1]);
}

/* tensor_create_ternary_from_hf_packed_2d: HF chunk-packed layout.
   For o=2, i_dim=2: hf_row_dim = (2+3)/4 = 1. Both output rows map to
   hf_byte_row 0; chunk = j (0 or 1) selects the 2-bit field. HF code v =
   code - 1, so code 1 -> 0, code 2 -> +1, code 0 -> -1. */
Test(torch_nn_quantization_bitlinear, hf_packed_load_then_forward) {
	/* Want decoded W row0 = [+1, -1], row1 = [0, +1].
	   row0 (chunk 0, bits 0-1): code = v+1 -> +1->2, -1->0.
	   row1 (chunk 1, bits 2-3): code = v+1 -> 0->1, +1->2.
	   byte[k=0]: chunk0 field = 2 (W[0][0]=+1), chunk1 field = 1 (W[1][0]=0)
	             -> 2 | (1<<2) = 0x06.
	   byte[k=1]: chunk0 field = 0 (W[0][1]=-1), chunk1 field = 2 (W[1][1]=+1)
	             -> 0 | (2<<2) = 0x08. */
	uint8_t hf[2] = {0x06, 0x08};
	TensorHandle W = tensor_create_ternary_from_hf_packed_2d(hf, /*o=*/2, /*i=*/2);
	double sd[] = {1.0, 1.0};
	TensorHandle scale = tensor_create_1d_f32(2, hcopy(sd, 2), 0);
	double xd[] = {10.0, 20.0};
	TensorHandle x = tensor_create_1d_f32(2, hcopy(xd, 2), 0);
	/* y0 = (+1)*10 + (-1)*20 = -10; y1 = 0*10 + 1*20 = 20. */
	TensorHandle y = tensor_bitlinear_fwd(W, scale, x, NULL);
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], -10.0, 1e-5, "hf-packed y0 exp -10 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 20.0, 1e-5, "hf-packed y1 exp 20 got %.6f", buf[1]);
}

/* tensor_absmean_per_row_2d on torch (real at::mean, unlike the tape impl
   that crashes on valid 2D input). row0 = [1,-2,3,-4] -> absmean 2.5;
   row1 = [0.5,-0.5,1.5,-1.5] -> absmean 1.0. */
Test(torch_nn_quantization_bitlinear, absmean_per_row) {
	double wd[] = {1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 1.5, -1.5};
	TensorHandle w = tensor_create_2d(2, 4, hcopy(wd, 8), /*rg=*/0);
	TensorHandle s = tensor_absmean_per_row_2d(w);
	cr_assert_eq(tensor_numel(s), 2, "absmean should yield one scalar per row");
	double buf[2];
	tensor_to_doubles(s, buf);
	cr_assert_float_eq(buf[0], 2.5, 1e-5, "absmean row0 exp 2.5 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 1.0, 1e-5, "absmean row1 exp 1.0 got %.6f", buf[1]);
}

/* tensor_ternary_quant_with_scale_2d on torch: round(W/scale), clamp to
   {-1,0,+1}, zero out rows whose original scale was 0. Verified via the
   resulting ternary W through tensor_bitlinear_fwd. */
Test(torch_nn_quantization_bitlinear, ternary_quant_with_scale_via_forward) {
	/* W [2,2] = [[2.0, -1.5], [0.4, -0.4]]; scale = [2.0, 0.0].
	   row0: round([1.0, -0.75]) = [1, -1] (clamp no-op).
	   row1: scale 0 -> active mask zeroes the row -> [0, 0]. */
	double wd[] = {2.0, -1.5, 0.4, -0.4};
	TensorHandle w = tensor_create_2d(2, 2, hcopy(wd, 4), /*rg=*/0);
	double sd[] = {2.0, 0.0};
	TensorHandle scale = tensor_create_1d_f32(2, hcopy(sd, 2), 0);
	TensorHandle Wt = tensor_ternary_quant_with_scale_2d(w, scale);
	/* Apply Wt via forward with unit scale and x = {10, 100}.
	   row0 = 1*10 + (-1)*100 = -90; row1 = 0*10 + 0*100 = 0. */
	double us[] = {1.0, 1.0};
	TensorHandle uscale = tensor_create_1d_f32(2, hcopy(us, 2), 0);
	double xd[] = {10.0, 100.0};
	TensorHandle x = tensor_create_1d_f32(2, hcopy(xd, 2), 0);
	TensorHandle y = tensor_bitlinear_fwd(Wt, uscale, x, NULL);
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], -90.0, 1e-5, "ternary-quant row0 exp -90 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 0.0, 1e-5, "ternary-quant zero-scale row exp 0 got %.6f", buf[1]);
}

/* ----------------------------------------------------------------------
   Own-line std::abort() guard coverage (deterministic libc abort, NOT a
   c10 throw — these SIGABRT death tests are reliable on torch).
   ---------------------------------------------------------------------- */

/* tensor_create_ternary_packed_2d byte-count mismatch (lines 43,46-47). */
Test(torch_nn_quantization_bitlinear, packed_byte_count_mismatch_aborts, .signal = SIGABRT) {
	uint8_t packed[2] = {0, 0};
	/* o=2,i=4 -> expected 2 bytes; pass 3 -> abort. */
	tensor_create_ternary_packed_2d(packed, /*packed_byte_count=*/3, /*o=*/2, /*i=*/4, /*rg=*/0);
}

/* tensor_create_ternary_packed_2d invalid 2-bit code 0x2 (lines 69,72-73).
   byte = 0x2 in slot 0 -> code 0x2 (none of 0/1/3) -> abort. */
Test(torch_nn_quantization_bitlinear, packed_invalid_code_aborts, .signal = SIGABRT) {
	uint8_t packed[2] = {0x02, 0x00}; /* o=2,i=4 -> 2 bytes; row0 slot0 code = 2. */
	tensor_create_ternary_packed_2d(packed, 2, /*o=*/2, /*i=*/4, /*rg=*/0);
}

/* tensor_create_ternary_packed_2d requires_grad=1 on int8 (lines 82,85). */
Test(torch_nn_quantization_bitlinear, packed_requires_grad_aborts, .signal = SIGABRT) {
	uint8_t packed[2] = {0, 0};
	tensor_create_ternary_packed_2d(packed, 2, /*o=*/2, /*i=*/4, /*requires_grad=*/1);
}

/* tensor_create_ternary_from_hf_packed_2d invalid HF code 3 (lines 170,173-174).
   chunk 0 (j=0) field = 3 -> v = 3-1 = 2 -> v>1 -> abort. */
Test(torch_nn_quantization_bitlinear, hf_packed_invalid_code_aborts, .signal = SIGABRT) {
	uint8_t hf[2] = {0x03, 0x00}; /* o=2,i=2; (row0,k0) chunk0 field = 3. */
	tensor_create_ternary_from_hf_packed_2d(hf, /*o=*/2, /*i=*/2);
}

/* tensor_absmean_per_row_2d non-2D input (lines 191,195-196). */
Test(torch_nn_quantization_bitlinear, absmean_rank1_aborts, .signal = SIGABRT) {
	double vd[] = {1.0, 2.0, 3.0};
	TensorHandle v = tensor_create_1d_f32(3, hcopy(vd, 3), /*rg=*/0);
	tensor_absmean_per_row_2d(v);
}

/* tensor_ternary_quant_with_scale_2d non-2D weight (lines 207-208,211-212). */
Test(torch_nn_quantization_bitlinear, ternary_quant_rank1_weight_aborts, .signal = SIGABRT) {
	double wd[] = {1.0, 2.0};
	TensorHandle w = tensor_create_1d_f32(2, hcopy(wd, 2), 0);
	double sd[] = {1.0, 1.0};
	TensorHandle scale = tensor_create_1d_f32(2, hcopy(sd, 2), 0);
	tensor_ternary_quant_with_scale_2d(w, scale);
}

/* tensor_ternary_quant_with_scale_2d scale-shape mismatch (lines 214-215,218-220). */
Test(torch_nn_quantization_bitlinear, ternary_quant_scale_mismatch_aborts, .signal = SIGABRT) {
	double wd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle w = tensor_create_2d(2, 2, hcopy(wd, 4), 0); /* o=2 */
	double sd[] = {1.0, 1.0, 1.0};                            /* scale size 3 != 2 */
	TensorHandle scale = tensor_create_1d_f32(3, hcopy(sd, 3), 0);
	tensor_ternary_quant_with_scale_2d(w, scale);
}

#endif /* BACKEND_TORCH */

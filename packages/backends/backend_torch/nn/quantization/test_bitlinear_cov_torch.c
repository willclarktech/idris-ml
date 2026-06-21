/* torch-only Criterion coverage suite for BitNet b1.58 quant edge arms.
 *
 * Complements test_bitlinear_torch.c by driving the dtype-variant arms of
 * the load-time quant path that the base suite only reaches at one dtype:
 *   - tensor_absmean_per_row_2d at F32 *and* F64 input (the at::mean +
 *     from_tensor dtype-propagation arm — output dtype tracks input).
 *   - tensor_ternary_quant_with_scale_2d at F32 *and* F64 weight+scale,
 *     including the upper clamp (ratio > 1 -> +1) and round arms, verified
 *     via a subsequent tensor_bitlinear_fwd.
 *   - tensor_bitlinear_fwd with an F64 scale + F64 activation (the base
 *     suite only exercises the F32 scale/x dequant path).
 *
 * torch CPU base dtype is F64; F32 tensors are made via the explicit
 * *_f32 creators. Values are chosen exactly representable in F32 so the
 * 1e-5 bar (mirroring the base suite's literal tolerances) holds.
 */

#include <criterion/criterion.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

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

/* tensor_absmean_per_row_2d on an F32 weight: at::mean(abs(w),1) keeps the
   input dtype, so the output is F32. row0 = [1,-2,3,-4] -> 2.5;
   row1 = [0.5,-0.5,1.5,-1.5] -> 1.0 (both exact in F32). */
Test(bitlinear_cov, absmean_per_row_f32) {
	double wd[] = {1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 1.5, -1.5};
	TensorHandle w = tensor_create_2d_f32(2, 4, hcopy(wd, 8), /*rg=*/0);
	TensorHandle s = tensor_absmean_per_row_2d(w);
	cr_assert_eq(tensor_numel(s), 2, "absmean should yield one scalar per row");
	cr_assert(strcmp(tensor_dtype_name(s), "F32") == 0, "absmean f32 input -> f32 output, got %s",
	          tensor_dtype_name(s));
	double buf[2];
	tensor_to_doubles(s, buf);
	cr_assert_float_eq(buf[0], 2.5, 1e-5, "absmean f32 row0 exp 2.5 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 1.0, 1e-5, "absmean f32 row1 exp 1.0 got %.6f", buf[1]);
}

/* tensor_absmean_per_row_2d on an explicit F64 weight: output dtype F64. */
Test(bitlinear_cov, absmean_per_row_f64) {
	double wd[] = {2.0, -4.0, 6.0, -8.0, 1.0, -1.0, 3.0, -3.0};
	TensorHandle w = tensor_create_2d_f64(2, 4, hcopy(wd, 8), /*rg=*/0);
	TensorHandle s = tensor_absmean_per_row_2d(w);
	cr_assert_eq(tensor_numel(s), 2, "absmean should yield one scalar per row");
	cr_assert(strcmp(tensor_dtype_name(s), "F64") == 0, "absmean f64 input -> f64 output, got %s",
	          tensor_dtype_name(s));
	double buf[2];
	tensor_to_doubles(s, buf);
	cr_assert_float_eq(buf[0], 5.0, 1e-12, "absmean f64 row0 exp 5.0 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 2.0, 1e-12, "absmean f64 row1 exp 2.0 got %.6f", buf[1]);
}

/* tensor_ternary_quant_with_scale_2d at F32 weight + F32 scale, exercising
   the upper-clamp arm (ratio 3 -> round 3 -> clamp +1). Verified via forward.
   W [2,2] = [[6,-6],[1,-0.4]]; scale = [2, 1].
   row0: ratio [3,-3] -> round [3,-3] -> clamp [+1,-1].
   row1: ratio [1,-0.4] -> round [1,0] -> clamp [+1,0]. */
Test(bitlinear_cov, ternary_quant_f32_upper_clamp) {
	double wd[] = {6.0, -6.0, 1.0, -0.4};
	TensorHandle w = tensor_create_2d_f32(2, 2, hcopy(wd, 4), /*rg=*/0);
	double sd[] = {2.0, 1.0};
	TensorHandle scale = tensor_create_1d_f32(2, hcopy(sd, 2), /*rg=*/0);
	TensorHandle Wt = tensor_ternary_quant_with_scale_2d(w, scale);
	/* Apply via forward with unit scale, x = {10, 100}.
	   row0 = 1*10 + (-1)*100 = -90; row1 = 1*10 + 0*100 = 10. */
	double us[] = {1.0, 1.0};
	TensorHandle uscale = tensor_create_1d_f32(2, hcopy(us, 2), 0);
	double xd[] = {10.0, 100.0};
	TensorHandle x = tensor_create_1d_f32(2, hcopy(xd, 2), 0);
	TensorHandle y = tensor_bitlinear_fwd(Wt, uscale, x, NULL);
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], -90.0, 1e-5, "ternary f32 row0 exp -90 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 10.0, 1e-5, "ternary f32 row1 exp 10 got %.6f", buf[1]);
}

/* tensor_ternary_quant_with_scale_2d at F64 weight + F64 scale, with the
   zero-scale row-mask arm (scale 0 -> active mask zeroes the row) and the
   upper-clamp arm together.
   W [2,2] = [[8,-8],[0.4,-0.4]]; scale = [2, 0].
   row0: ratio [4,-4] -> round [4,-4] -> clamp [+1,-1].
   row1: scale 0 -> safe-clamped divisor, then active mask -> [0,0]. */
Test(bitlinear_cov, ternary_quant_f64_clamp_and_zero_row) {
	double wd[] = {8.0, -8.0, 0.4, -0.4};
	TensorHandle w = tensor_create_2d_f64(2, 2, hcopy(wd, 4), /*rg=*/0);
	double sd[] = {2.0, 0.0};
	TensorHandle scale = tensor_create_1d_f64(2, hcopy(sd, 2), /*rg=*/0);
	TensorHandle Wt = tensor_ternary_quant_with_scale_2d(w, scale);
	double us[] = {1.0, 1.0};
	TensorHandle uscale = tensor_create_1d_f64(2, hcopy(us, 2), 0);
	double xd[] = {10.0, 100.0};
	TensorHandle x = tensor_create_1d_f64(2, hcopy(xd, 2), 0);
	/* row0 = 1*10 + (-1)*100 = -90; row1 (masked) = 0. */
	TensorHandle y = tensor_bitlinear_fwd(Wt, uscale, x, NULL);
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], -90.0, 1e-12, "ternary f64 row0 exp -90 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 0.0, 1e-12, "ternary f64 zero-scale row exp 0 got %.6f", buf[1]);
}

/* tensor_bitlinear_fwd with an F64 scale + F64 activation (W.to(F64) dequant
   path). W [2,2] r0 = [+1, 0], r1 = [0, +1]; scale = {2, 3}; x = {5, 7}.
   y0 = 2*(1*5) = 10; y1 = 3*(1*7) = 21. */
Test(bitlinear_cov, forward_f64_scale_and_act) {
	int r0[] = {1, 0};
	int r1[] = {0, 1};
	uint8_t packed[2]; /* i=2 -> bytes_per_row=1, 2 rows. */
	pack_row(r0, 2, &packed[0]);
	pack_row(r1, 2, &packed[1]);
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 2, /*o=*/2, /*i=*/2, /*rg=*/0);
	double sd[] = {2.0, 3.0};
	TensorHandle scale = tensor_create_1d_f64(2, hcopy(sd, 2), 0);
	double xd[] = {5.0, 7.0};
	TensorHandle x = tensor_create_1d_f64(2, hcopy(xd, 2), 0);
	TensorHandle y = tensor_bitlinear_fwd(W, scale, x, NULL);
	cr_assert(strcmp(tensor_dtype_name(y), "F64") == 0, "f64 forward -> f64 output, got %s",
	          tensor_dtype_name(y));
	double buf[2];
	tensor_to_doubles(y, buf);
	cr_assert_float_eq(buf[0], 10.0, 1e-12, "f64 fwd y0 exp 10 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 21.0, 1e-12, "f64 fwd y1 exp 21 got %.6f", buf[1]);
}

#endif /* BACKEND_TORCH */

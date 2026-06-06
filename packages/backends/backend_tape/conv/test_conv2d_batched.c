/* Criterion suite for tape `tensor_conv2d_batched`.
 *
 * input [B=1, inC=1, H=2, W=2] = [[1,2],[3,4]] (single sample),
 * kernel [outC=1, inC=1, kH=2, kW=2] = [[1,1],[1,1]], no bias, pad=0, stride=1.
 * Forward: B=1 reduces to single-sample conv2d; out[0,0,0,0] = 1+2+3+4 = 10.
 * Backward sum-loss: d_in[i] = 1, d_k[i] = in[i].
 *
 * RED: dispatch NULL → d_in[0] expected 1 fires.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

#ifdef BACKEND_TAPE
/* Dtag values mirroring DType.Core ("13/14/15=F16/F32/F64"). */
#define DTAG_F32 14
#endif

Test(conv_conv2d_batched, forward_and_backward) {
    param_clear();
    double in_data[4] = {1.0, 2.0, 3.0, 4.0};
    double k_data[4]  = {1.0, 1.0, 1.0, 1.0};
    int sh_in[4] = {1, 1, 2, 2};
    int sh_k[4]  = {1, 1, 2, 2};
    TensorHandle in = tensor_create(in_data, sh_in, 4, 1);
    TensorHandle k  = tensor_create(k_data,  sh_k,  4, 1);
    param_register("in", in);
    param_register("k",  k);

    TensorHandle out = tensor_conv2d_batched(in, k, (TensorHandle)0, 0, 0, 1, 1);
    cr_assert_float_eq(tensor_item_1d(out, 0), 10.0, 1e-12);

    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12, "d_in[%d]", i);
    double exp_k[4] = {1.0, 2.0, 3.0, 4.0};
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(param_grad_item_at(1, i), exp_k[i], 1e-12, "d_k[%d]", i);
}

#ifdef BACKEND_TAPE
/* Paired F32 vs F64 forward on a realistic CNN shape — input [B=2, inC=3, H=8, W=8],
   kernel [outC=4, inC=3, kH=3, kW=3]. Verifies the new im2col + cblas_sgemm
   path matches the F64 cblas_dgemm reference within F32 precision.

   Values chosen to be exactly representable in F32 (small integers / halves)
   so the F32 vs F64 comparison is bounded by the BLAS accumulation error
   (~1e-6 relative for sgemm on K=27 dot products) rather than input
   quantization. */
Test(conv_conv2d_batched, f32_forward_matches_f64_reference) {
    param_clear();
    int B = 2, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
    int in_numel = B * inC * H * W;
    int k_numel  = outC * inC * kH * kW;

    /* Deterministic input + kernel; values land in F32-exact range. */
    double* in_data  = (double*)malloc(in_numel * sizeof(double));
    double* k_data   = (double*)malloc(k_numel * sizeof(double));
    for (int i = 0; i < in_numel; i++) in_data[i]  = ((i % 11) - 5) * 0.25;
    for (int i = 0; i < k_numel;  i++) k_data[i]   = ((i % 7)  - 3) * 0.125;

    int sh_in[4] = {B, inC, H, W};
    int sh_k[4]  = {outC, inC, kH, kW};

    /* F64 reference: tensor_create copies; we keep our buffer alive. */
    double* in_f64 = (double*)malloc(in_numel * sizeof(double));
    double* k_f64  = (double*)malloc(k_numel * sizeof(double));
    memcpy(in_f64, in_data, in_numel * sizeof(double));
    memcpy(k_f64,  k_data,  k_numel  * sizeof(double));
    TensorHandle in64 = tensor_create(in_f64, sh_in, 4, 0);
    TensorHandle k64  = tensor_create(k_f64,  sh_k,  4, 0);
    free(in_f64); free(k_f64);
    TensorHandle out64 = tensor_conv2d_batched(in64, k64, (TensorHandle)0, 0, 0, 1, 1);

    /* F32 candidate: construct via streamed-dtag entry point. */
    double* in_f32 = (double*)malloc(in_numel * sizeof(double));
    double* k_f32  = (double*)malloc(k_numel  * sizeof(double));
    memcpy(in_f32, in_data, in_numel * sizeof(double));
    memcpy(k_f32,  k_data,  k_numel  * sizeof(double));
    TensorHandle in32 = tensor_create_streamed(in_f32, sh_in, 4, 0, 0, DTAG_F32);
    TensorHandle k32  = tensor_create_streamed(k_f32,  sh_k,  4, 0, 0, DTAG_F32);
    free(in_f32); free(k_f32);
    TensorHandle out32 = tensor_conv2d_batched(in32, k32, (TensorHandle)0, 0, 0, 1, 1);

    cr_assert_str_eq(tensor_dtype_name(out32), "F32",
        "F32 input should produce F32 output (got %s)",
        tensor_dtype_name(out32));

    int out_numel = B * outC * (H - kH + 1) * (W - kW + 1);
    double* buf64 = (double*)malloc(out_numel * sizeof(double));
    double* buf32 = (double*)malloc(out_numel * sizeof(double));
    tensor_to_doubles(out64, buf64);
    tensor_to_doubles(out32, buf32);

    /* sgemm accumulates in F32 mantissa; tolerance covers ~K=27 fmadds. */
    for (int i = 0; i < out_numel; i++) {
        double absdiff = buf32[i] - buf64[i];
        if (absdiff < 0) absdiff = -absdiff;
        double tol = 1e-4 * (buf64[i] < 0 ? -buf64[i] : buf64[i]) + 1e-5;
        cr_assert(absdiff < tol,
            "out[%d]: F32 %.7f vs F64 %.7f (diff %.2e, tol %.2e)",
            i, buf32[i], buf64[i], absdiff, tol);
    }

    free(buf64); free(buf32);
    free(in_data); free(k_data);
}
#endif

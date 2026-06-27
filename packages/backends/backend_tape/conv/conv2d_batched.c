/* conv/conv2d_batched.c — batched 2D convolution (forward + backward).
 *
 * Input [B, inC, H, W], kernel [outC, inC, kH, kW], bias
 * [outC] or NULL. Output [B, outC, oH, oW].
 *
 * Forward uses the standard im2col + cblas_dgemm decomposition:
 *     X_col [M, K] where M = B*oH*oW, K = inC*kH*kW
 *     Y_unf [M, outC] = X_col @ W^T   (single dgemm)
 *     out   [B, outC, oH, oW] = permute(Y_unf, (0,2,1)) + bias broadcast
 * This is what PyTorch / cuDNN do at the unfused-conv path; the dgemm
 * replaces an O(B·outC·inC·kH·kW·oH·oW) hand-rolled triple loop with
 * Apple Accelerate's blocked sgemm.
 *
 * The F32 forward path mirrors the F64 path via the parallel
 * `conv2d_im2col_f32` helper + `cblas_sgemm`. The F32 backward
 * keeps widening to double buffers so the existing dgemm path
 * covers it without a separate sgemm implementation.
 *
 * Conv2DBatchedMeta stays in tape.h; im2col / col2im helpers are TU-
 * private to this file (no other op needs them).
 */

#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // IWYU pragma: keep — umbrella; provides cblas_* + Cblas*
#endif
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../training/autograd/op_dispatch.h"
#include "../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

/* im2col: build X_col [M, K] where M = B*oH*oW, K = inC*kH*kW.
   Each row is one (batch, out-position)'s unfolded inC*kH*kW window. */
static void conv2d_im2col(const double* input, int B, int inC, int H, int W, int kH, int kW,
                          int padH, int padW, int strideH, int strideW, int oH, int oW,
                          double* X_col) {
	int K = inC * kH * kW;
	int M = B * oH * oW;
	memset(X_col, 0, (size_t)M * K * sizeof(double));
	for (int b = 0; b < B; b++) {
		const double* inp_b = input + (size_t)b * inC * H * W;
		for (int oh = 0; oh < oH; oh++) {
			for (int ow = 0; ow < oW; ow++) {
				double* row = X_col + ((size_t)b * oH * oW + (size_t)oh * oW + ow) * K;
				for (int ic = 0; ic < inC; ic++) {
					for (int kh = 0; kh < kH; kh++) {
						int ih = oh * strideH - padH + kh;
						if (ih < 0 || ih >= H) continue;
						for (int kw = 0; kw < kW; kw++) {
							int iw = ow * strideW - padW + kw;
							if (iw < 0 || iw >= W) continue;
							row[ic * kH * kW + kh * kW + kw] = inp_b[ic * H * W + ih * W + iw];
						}
					}
				}
			}
		}
	}
}

/* im2col F32 sibling: identical structure to conv2d_im2col but with
   float-typed buffers, for the F32 forward sgemm path. */
static void conv2d_im2col_f32(const float* input, int B, int inC, int H, int W, int kH, int kW,
                              int padH, int padW, int strideH, int strideW, int oH, int oW,
                              float* X_col) {
	int K = inC * kH * kW;
	int M = B * oH * oW;
	memset(X_col, 0, (size_t)M * K * sizeof(float));
	for (int b = 0; b < B; b++) {
		const float* inp_b = input + (size_t)b * inC * H * W;
		for (int oh = 0; oh < oH; oh++) {
			for (int ow = 0; ow < oW; ow++) {
				float* row = X_col + ((size_t)b * oH * oW + (size_t)oh * oW + ow) * K;
				for (int ic = 0; ic < inC; ic++) {
					for (int kh = 0; kh < kH; kh++) {
						int ih = oh * strideH - padH + kh;
						if (ih < 0 || ih >= H) continue;
						for (int kw = 0; kw < kW; kw++) {
							int iw = ow * strideW - padW + kw;
							if (iw < 0 || iw >= W) continue;
							row[ic * kH * kW + kh * kW + kw] = inp_b[ic * H * W + ih * W + iw];
						}
					}
				}
			}
		}
	}
}

/* col2im (gradient accumulating version): scatter dX_col [M, K] back into
   dInput [B, inC, H, W]. Padding cells are dropped. */
static void conv2d_col2im_accumulate(const double* dX_col, int B, int inC, int H, int W, int kH,
                                     int kW, int padH, int padW, int strideH, int strideW, int oH,
                                     int oW, double* dInput) {
	int K = inC * kH * kW;
	for (int b = 0; b < B; b++) {
		double* din_b = dInput + (size_t)b * inC * H * W;
		for (int oh = 0; oh < oH; oh++) {
			for (int ow = 0; ow < oW; ow++) {
				const double* row = dX_col + ((size_t)b * oH * oW + (size_t)oh * oW + ow) * K;
				for (int ic = 0; ic < inC; ic++) {
					for (int kh = 0; kh < kH; kh++) {
						int ih = oh * strideH - padH + kh;
						if (ih < 0 || ih >= H) continue;
						for (int kw = 0; kw < kW; kw++) {
							int iw = ow * strideW - padW + kw;
							if (iw < 0 || iw >= W) continue;
							din_b[ic * H * W + ih * W + iw] += row[ic * kH * kW + kh * kW + kw];
						}
					}
				}
			}
		}
	}
}

TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                   int padH, int padW, int strideH, int strideW) {
	Tensor* input = (Tensor*)hinput;
	Tensor* kernel = (Tensor*)hkernel;
	Tensor* bias = (Tensor*)hbias;
	if (input->dtype_tag != kernel->dtype_tag || (bias && bias->dtype_tag != input->dtype_tag))
		tape_abort_mixed_dtype("tensor_conv2d_batched");

	int B = input->shape[0], inC = input->shape[1];
	int H = input->shape[2], W = input->shape[3];
	int outC = kernel->shape[0], kH = kernel->shape[2], kW = kernel->shape[3];
	int oH = (H + 2 * padH - kH) / strideH + 1;
	int oW = (W + 2 * padW - kW) / strideW + 1;
	int out_numel = B * outC * oH * oW;
	int is_f32 = (input->dtype_tag == DT_F32);
	int out_shape[] = {B, outC, oH, oW};
	int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);

	void* out_buf;
	if (is_f32) {
		/* F32 forward: im2col + cblas_sgemm, mirroring the F64 path.
		   F32 BLAS is already wired (mm.c, bmm.c, linear_2d.c). */
		int K = inC * kH * kW;
		int M = B * oH * oW;
		float* X_col = (float*)calloc((size_t)M * K, sizeof(float));
		conv2d_im2col_f32((const float*)input->data, B, inC, H, W, kH, kW, padH, padW, strideH,
		                  strideW, oH, oW, X_col);
		float* Y_unf = (float*)calloc((size_t)M * outC, sizeof(float));
#ifdef __APPLE__
		// NOLINTNEXTLINE(misc-include-cleaner): BLAS symbols via Accelerate umbrella
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, outC, K, 1.0f, X_col, K,
		            (const float*)kernel->data, K, 0.0f, Y_unf, outC);
#else
		for (int m = 0; m < M; m++)
			for (int oc = 0; oc < outC; oc++) {
				float s = 0.0f;
				for (int k = 0; k < K; k++)
					s += X_col[m * K + k] * ((const float*)kernel->data)[oc * K + k];
				Y_unf[m * outC + oc] = s;
			}
#endif
		float* out = arena_alloc(out_numel * sizeof(float));
		for (int b = 0; b < B; b++) {
			for (int oc = 0; oc < outC; oc++) {
				float b_val = bias ? (float)tape_load_d(bias, oc) : 0.0f;
				float* out_chan = out + ((size_t)b * outC + oc) * oH * oW;
				for (int oh = 0; oh < oH; oh++) {
					for (int ow = 0; ow < oW; ow++) {
						int row = b * oH * oW + oh * oW + ow;
						// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
						out_chan[oh * oW + ow] = Y_unf[row * outC + oc] + b_val;
					}
				}
			}
		}
		free(Y_unf);
		free(X_col);
		out_buf = out;
	} else {
		int K = inC * kH * kW;
		int M = B * oH * oW;
		double* X_col = (double*)calloc((size_t)M * K, sizeof(double));
		conv2d_im2col(input->data, B, inC, H, W, kH, kW, padH, padW, strideH, strideW, oH, oW,
		              X_col);
		double* Y_unf = calloc((size_t)M * outC, sizeof(double));
#ifdef __APPLE__
		// NOLINTNEXTLINE(misc-include-cleaner): BLAS symbols via Accelerate umbrella
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, outC, K, 1.0, X_col, K,
		            kernel->data, K, 0.0, Y_unf, outC);
#else
		for (int m = 0; m < M; m++)
			for (int oc = 0; oc < outC; oc++) {
				double s = 0;
				for (int k = 0; k < K; k++)
					s += X_col[m * K + k] * ((double*)kernel->data)[oc * K + k];
				Y_unf[m * outC + oc] = s;
			}
#endif
		double* out = calloc(out_numel, sizeof(double));
		for (int b = 0; b < B; b++) {
			for (int oc = 0; oc < outC; oc++) {
				double b_val = bias ? ((double*)bias->data)[oc] : 0.0;
				double* out_chan = out + ((size_t)b * outC + oc) * oH * oW;
				for (int oh = 0; oh < oH; oh++) {
					for (int ow = 0; ow < oW; ow++) {
						int row = b * oH * oW + oh * oW + ow;
						// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
						out_chan[oh * oW + ow] = Y_unf[row * outC + oc] + b_val;
					}
				}
			}
		}
		free(Y_unf);
		free(X_col);
		out_buf = out;
	}

	Tensor* r;
	if (is_f32)
		r = make_tensor_arena_f32((float*)out_buf, out_numel, out_shape, 4, rg);
	else {
		r = make_tensor((double*)out_buf, out_shape, 4, rg);
		free(out_buf);
	}

	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_CONV2D_BATCHED, r, input, kernel, 0);
		Conv2DBatchedMeta* meta = arena_alloc(sizeof(Conv2DBatchedMeta));
		meta->B = B;
		meta->inC = inC;
		meta->outC = outC;
		meta->H = H;
		meta->W = W;
		meta->kH = kH;
		meta->kW = kW;
		meta->padH = padH;
		meta->padW = padW;
		meta->strH = strideH;
		meta->strW = strideW;
		meta->oH = oH;
		meta->oW = oW;
		e->op_meta = meta;
		/* Store bias pointer in scalar_arg slot (cast) for backward */
		e->inputs = (Tensor**)bias;
	}
	return r;
}

static void tape_backward_conv2d_batched(TapeEntry* e) {
	/* r = conv2d_batched(input [B,inC,H,W], kernel [outC,inC,kH,kW]) + bias
	   r=[B,outC,oH,oW]. Backward via im2col + cblas_dgemm in F64. For F32
	   tensors the F64 dgemm path is reused by widening *data* (input/kernel)
	   to temporary double buffers — but the GRAD buffers of F32 tensors are
	   float-sized too (see ensure_grad), so every grad read/write must go
	   through the typed accessors: the result grad is widened into `rgrad`,
	   and the input/kernel grads are written via a double scratch then
	   scattered with tape_grad_add_d. (Direct ((double*)t->grad)[i] on an
	   F32 tensor reads/writes 8 bytes into a 4-byte slot — a heap overflow.) */
	Conv2DBatchedMeta* meta = (Conv2DBatchedMeta*)e->op_meta;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	Tensor* r = e->result;
	int B = meta->B;
	int inC = meta->inC, outC = meta->outC;
	int HH = meta->H, WW = meta->W, kH = meta->kH, kW = meta->kW;
	int padH = meta->padH, padW = meta->padW;
	int strideH = meta->strH, strideW = meta->strW;
	int oH = meta->oH, oW = meta->oW;
	int K_unf = inC * kH * kW;
	int M_unf = B * oH * oW;
	int out_per_sample = outC * oH * oW;
	ensure_grad(r);

	Tensor* bias_t = (Tensor*)e->inputs;
	/* d/dW needs X (a) via im2col, d/dX needs W (b) via dgemm — both
	 * inputs required for either non-bias gradient. */
	int need_dW = a && b && b->requires_grad;
	int need_dX = a && b && a->requires_grad;
	int need_dB = bias_t && bias_t->requires_grad;

	/* For F32 inputs/kernel, widen to double buffers so the existing
	   cblas_dgemm + conv2d_im2col paths work unchanged. */
	double* a_data_dbl = NULL;
	double* b_data_dbl = NULL;
	const void* a_data_ptr = a ? a->data : NULL;
	const void* b_data_ptr = b ? b->data : NULL;
	if (a && a->dtype_tag == DT_F32) {
		size_t a_n = (size_t)B * inC * HH * WW;
		a_data_dbl = (double*)malloc(a_n * sizeof(double));
		for (size_t i = 0; i < a_n; i++)
			a_data_dbl[i] = (double)((float*)a->data)[i];
		a_data_ptr = a_data_dbl;
	}
	if (b && b->dtype_tag == DT_F32) {
		size_t b_n = (size_t)outC * inC * kH * kW;
		b_data_dbl = (double*)malloc(b_n * sizeof(double));
		for (size_t i = 0; i < b_n; i++)
			b_data_dbl[i] = (double)((float*)b->data)[i];
		b_data_ptr = b_data_dbl;
	}

	/* r->grad may be F32 storage (float-sized); the F64 dgemm/permute below
	   assume double. Widen the result grad through the typed accessor into a
	   double scratch buffer. */
	size_t r_n = (size_t)B * out_per_sample;
	double* rgrad = (double*)malloc(r_n * sizeof(double));
	for (size_t i = 0; i < r_n; i++)
		rgrad[i] = tape_grad_load_d(r, (int)i);

	/* Permute dY [B, outC, oH, oW] -> dY_unf [B*oH*oW, outC] */
	double* dY_unf =
	    (need_dW || need_dX) ? (double*)calloc((size_t)M_unf * outC, sizeof(double)) : NULL;
	if (dY_unf) {
		for (int bb = 0; bb < B; bb++) {
			const double* dout_b = rgrad + (size_t)bb * out_per_sample;
			for (int oc = 0; oc < outC; oc++) {
				for (int oh = 0; oh < oH; oh++) {
					for (int ow = 0; ow < oW; ow++) {
						int row = bb * oH * oW + oh * oW + ow;
						/* row<M_unf, oc<outC: provably in bounds */
						// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
						dY_unf[row * outC + oc] = dout_b[oc * oH * oW + oh * oW + ow];
					}
				}
			}
		}
	}

	/* d_kernel — single dgemm: dW[outC,K] = dY_unf^T[outC,M] @ X_col[M,K] */
	if (need_dW && dY_unf) {
		ensure_grad(b);
		double* X_col = (double*)calloc((size_t)M_unf * K_unf, sizeof(double));
		conv2d_im2col((const double*)a_data_ptr, B, inC, HH, WW, kH, kW, padH, padW, strideH,
		              strideW, oH, oW, X_col);
#ifdef __APPLE__
		/* dgemm into a double scratch (beta=0), then scatter through the
		   typed accessor so an F32 kernel's float-sized grad isn't overrun. */
		double* dW_tmp = (double*)calloc((size_t)outC * K_unf, sizeof(double));
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, outC, K_unf, M_unf, 1.0, dY_unf, outC,
		            X_col, K_unf, 0.0, dW_tmp, K_unf);
		for (int i = 0; i < outC * K_unf; i++)
			tape_grad_add_d(b, i, dW_tmp[i]);
		free(dW_tmp);
#else
		for (int oc = 0; oc < outC; oc++)
			for (int kk = 0; kk < K_unf; kk++) {
				double s = 0;
				for (int m = 0; m < M_unf; m++)
					s += dY_unf[m * outC + oc] * X_col[m * K_unf + kk];
				tape_grad_add_d(b, oc * K_unf + kk, s);
			}
#endif
		free(X_col);
	}

	/* d_input — dX_col[M,K] = dY_unf[M,outC] @ W[outC,K], then col2im */
	if (need_dX && dY_unf) {
		ensure_grad(a);
		double* dX_col = calloc((size_t)M_unf * K_unf, sizeof(double));
#ifdef __APPLE__
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_unf, K_unf, outC, 1.0, dY_unf,
		            outC, (const double*)b_data_ptr, K_unf, 0.0, dX_col, K_unf);
#else
		for (int m = 0; m < M_unf; m++)
			for (int kk = 0; kk < K_unf; kk++) {
				double s = 0;
				for (int oc = 0; oc < outC; oc++)
					// NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult,clang-analyzer-security.ArrayBound)
					s += dY_unf[m * outC + oc] * ((const double*)b_data_ptr)[oc * K_unf + kk];
				dX_col[m * K_unf + kk] = s;
			}
#endif
		/* col2im into a double scratch, then scatter through the typed
		   accessor so an F32 input's float-sized grad isn't overrun. */
		size_t a_n = (size_t)B * inC * HH * WW;
		double* dA_tmp = (double*)calloc(a_n, sizeof(double));
		conv2d_col2im_accumulate(dX_col, B, inC, HH, WW, kH, kW, padH, padW, strideH, strideW, oH,
		                         oW, dA_tmp);
		for (size_t i = 0; i < a_n; i++)
			tape_grad_add_d(a, (int)i, dA_tmp[i]);
		free(dA_tmp);
		free(dX_col);
	}

	/* d_bias — sum across B and (oH, oW) per output channel */
	if (need_dB) {
		ensure_grad(bias_t);
		for (int oc = 0; oc < outC; oc++) {
			double s = 0;
			for (int bb = 0; bb < B; bb++) {
				const double* dout_b = rgrad + (size_t)bb * out_per_sample;
				for (int oh = 0; oh < oH; oh++)
					for (int ow = 0; ow < oW; ow++)
						/* oc<outC, oh<oH, ow<oW: provably in bounds */
						// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
						s += dout_b[oc * oH * oW + oh * oW + ow];
			}
			tape_grad_add_d(bias_t, oc, s);
		}
	}
	if (dY_unf) free(dY_unf);
	if (a_data_dbl) free(a_data_dbl);
	if (b_data_dbl) free(b_data_dbl);
	free(rgrad);
}

TAPE_REGISTER_OP(OP_CONV2D_BATCHED, tape_backward_conv2d_batched)

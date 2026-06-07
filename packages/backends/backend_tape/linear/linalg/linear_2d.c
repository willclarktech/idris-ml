/* linear/linalg/linear_2d.c — fused batched linear (forward + backward).
 *
 * F64 BIT-EXACT RISK — cblas_dgemm arg order & transpose
 * flags. Y[B,o] = X[B,i] @ W[o,i]^T + bias[o]. Caches X as double* in
 * Linear2dMeta. Backward:
 *   dW    = dY^T @ X
 *   dX    = dY @ W
 *   dbias = sum_b dY[b, :]
 */

#include <string.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_linear_2d(TensorHandle hW, TensorHandle hX, TensorHandle hbias) {
	Tensor* W = (Tensor*)hW;
	Tensor* X = (Tensor*)hX;
	Tensor* bias = (Tensor*)hbias;
	if (W->dtype_tag != X->dtype_tag || (bias && bias->dtype_tag != W->dtype_tag))
		tape_abort_mixed_dtype("tensor_linear_2d");
	int oo = W->shape[0], ii = W->shape[1];
	int BB = X->shape[0];
	int out_shape[] = {BB, oo};
	int rg = W->requires_grad || X->requires_grad || (bias && bias->requires_grad);

	/* Zero-dim guard (see mm.c). cblas_*gemm rejects lda=0 when ii=0. */
	if (BB == 0 || oo == 0 || ii == 0) {
		Tensor* r = tape_zero_tensor(out_shape, 2, W->dtype_tag, rg);
		if (bias && BB > 0 && oo > 0) {
			/* ii=0 case: matmul drops out but bias broadcasts across batch */
			if (W->dtype_tag == DT_F32) {
				for (int b = 0; b < BB; b++)
					for (int o = 0; o < oo; o++)
						((float*)r->data)[b * oo + o] = ((float*)bias->data)[o];
			} else {
				for (int b = 0; b < BB; b++)
					for (int o = 0; o < oo; o++)
						((double*)r->data)[b * oo + o] = ((double*)bias->data)[o];
			}
		}
		return r;
	}

	if (W->dtype_tag == DT_F32) {
		float* out_data = arena_alloc((size_t)BB * oo * sizeof(float));
#ifdef __APPLE__
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, BB, oo, ii, 1.0f,
		            (const float*)X->data, ii, (const float*)W->data, ii, 0.0f, out_data, oo);
#else
		for (int b = 0; b < BB; b++) {
			for (int o = 0; o < oo; o++) {
				float s = 0;
				for (int j = 0; j < ii; j++)
					s += ((float*)X->data)[(size_t)b * ii + j] *
					     ((float*)W->data)[(size_t)o * ii + j];
				out_data[(size_t)b * oo + o] = s;
			}
		}
#endif
		if (bias) {
			for (int b = 0; b < BB; b++) {
#ifdef __APPLE__
				vDSP_vadd(out_data + (size_t)b * oo, 1, (const float*)bias->data, 1,
				          out_data + (size_t)b * oo, 1, (vDSP_Length)oo);
#else
				for (int o = 0; o < oo; o++)
					out_data[(size_t)b * oo + o] += ((float*)bias->data)[o];
#endif
			}
		}
		Tensor* r = make_tensor_arena_f32(out_data, BB * oo, out_shape, 2, rg);
		if (rg) {
			TapeEntry* e = tape_append(OP_LINEAR_2D, r, W, X, 0);
			Linear2dMeta* meta = arena_alloc(sizeof(Linear2dMeta));
			meta->B = BB;
			meta->i = ii;
			meta->o = oo;
			meta->x_vals = arena_alloc((size_t)BB * ii * sizeof(double));
			for (int j = 0; j < BB * ii; j++)
				meta->x_vals[j] = (double)((float*)X->data)[j];
			meta->bias = bias;
			e->op_meta = meta;
		}
		return r;
	}

	double* out_data = malloc((size_t)BB * oo * sizeof(double));
#ifdef __APPLE__
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, BB, oo, ii, 1.0, X->data, ii, W->data, ii,
	            0.0, out_data, oo);
#else
	for (int b = 0; b < BB; b++) {
		for (int o = 0; o < oo; o++) {
			double s = 0;
			for (int j = 0; j < ii; j++)
				s +=
				    ((double*)X->data)[(size_t)b * ii + j] * ((double*)W->data)[(size_t)o * ii + j];
			out_data[(size_t)b * oo + o] = s;
		}
	}
#endif
	if (bias) {
		for (int b = 0; b < BB; b++) {
#ifdef __APPLE__
			vDSP_vaddD(out_data + (size_t)b * oo, 1, bias->data, 1, out_data + (size_t)b * oo, 1,
			           (vDSP_Length)oo);
#else
			for (int o = 0; o < oo; o++)
				out_data[(size_t)b * oo + o] += ((double*)bias->data)[o];
#endif
		}
	}
	Tensor* r = make_tensor(out_data, out_shape, 2, rg);
	free(out_data);
	if (rg) {
		TapeEntry* e = tape_append(OP_LINEAR_2D, r, W, X, 0);
		Linear2dMeta* meta = arena_alloc(sizeof(Linear2dMeta));
		meta->B = BB;
		meta->i = ii;
		meta->o = oo;
		meta->x_vals = arena_alloc((size_t)BB * ii * sizeof(double));
		memcpy(meta->x_vals, X->data, (size_t)BB * ii * sizeof(double));
		meta->bias = bias;
		e->op_meta = meta;
	}
	return r;
}

static void tape_backward_linear_2d(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1; /* W */
	Tensor* b = e->arg2; /* X */
	Linear2dMeta* lm2 = (Linear2dMeta*)e->op_meta;
	int B2 = lm2->B, i2 = lm2->i, o2 = lm2->o;
	double* x_vals_2 = lm2->x_vals;
	int is_f32 = (a->dtype_tag == DT_F32);
	ensure_grad(r);
	if (a->requires_grad) {
		ensure_grad(a);
		if (is_f32) {
			for (int oo = 0; oo < o2; oo++)
				for (int jj = 0; jj < i2; jj++) {
					double s = 0;
					for (int bb = 0; bb < B2; bb++)
						s += tape_grad_load_d(r, bb * o2 + oo) * x_vals_2[bb * i2 + jj];
					tape_grad_add_d(a, oo * i2 + jj, s);
				}
		} else {
#ifdef __APPLE__
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, o2, i2, B2, 1.0, r->grad, o2,
			            x_vals_2, i2, 1.0, a->grad, i2);
#else
			for (int oo = 0; oo < o2; oo++)
				for (int jj = 0; jj < i2; jj++) {
					double s = 0;
					for (int bb = 0; bb < B2; bb++)
						s += tape_grad_load_d(r, bb * o2 + oo) * x_vals_2[bb * i2 + jj];
					tape_grad_add_d(a, oo * i2 + jj, s);
				}
#endif
		}
	}
	if (b && b->requires_grad) {
		ensure_grad(b);
		if (is_f32) {
			for (int bb = 0; bb < B2; bb++)
				for (int jj = 0; jj < i2; jj++) {
					double s = 0;
					for (int oo = 0; oo < o2; oo++)
						s += tape_grad_load_d(r, bb * o2 + oo) * tape_load_d(a, oo * i2 + jj);
					tape_grad_add_d(b, bb * i2 + jj, s);
				}
		} else {
#ifdef __APPLE__
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B2, i2, o2, 1.0, r->grad, o2,
			            a->data, i2, 1.0, b->grad, i2);
#else
			for (int bb = 0; bb < B2; bb++)
				for (int jj = 0; jj < i2; jj++) {
					double s = 0;
					for (int oo = 0; oo < o2; oo++)
						s += tape_grad_load_d(r, bb * o2 + oo) * ((double*)a->data)[oo * i2 + jj];
					tape_grad_add_d(b, bb * i2 + jj, s);
				}
#endif
		}
	}
	if (lm2->bias && lm2->bias->requires_grad) {
		ensure_grad(lm2->bias);
		for (int oo = 0; oo < o2; oo++) {
			double s = 0;
			for (int bb = 0; bb < B2; bb++)
				s += tape_grad_load_d(r, bb * o2 + oo);
			tape_grad_add_d(lm2->bias, oo, s);
		}
	}
}

TAPE_REGISTER_OP(OP_LINEAR_2D, tape_backward_linear_2d)

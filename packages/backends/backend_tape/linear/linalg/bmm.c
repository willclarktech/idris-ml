/* linear/linalg/bmm.c — batched matrix-matrix multiply (shared b).
 *
 * r[bi] = a[bi] @ b where a=[B,m,n], b=[n,k], r=[B,m,k].
 * b is shared across the batch. F64 bit-exact via cblas_dgemm.
 * Backward collapses [B,m,*] to [B*m,*] for the shared d_b update.
 */

#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // IWYU pragma: keep — umbrella; provides cblas_* + Cblas*
#endif
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_bmm(TensorHandle ha, TensorHandle hb) {
	Tensor* a = (Tensor*)ha;
	Tensor* b = (Tensor*)hb;
	if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_bmm");
	int B = a->shape[0], m = a->shape[1], n = a->shape[2], k = b->shape[1];
	int rg = a->requires_grad || b->requires_grad;
	int shape[] = {B, m, k};
	/* Zero-dim guard (see mm.c). Per-batch cblas_*gemm rejects lda=0. */
	if (B == 0 || m == 0 || n == 0 || k == 0) return tape_zero_tensor(shape, 3, a->dtype_tag, rg);
	if (a->dtype_tag == DT_F32) {
		float* data = arena_alloc((size_t)B * m * k * sizeof(float));
		for (int bi = 0; bi < B; bi++) {
#ifdef __APPLE__
			// NOLINTNEXTLINE(misc-include-cleaner): BLAS symbols via Accelerate umbrella
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0f,
			            ((const float*)a->data) + (size_t)bi * m * n, n, (const float*)b->data, k,
			            0.0f, data + (size_t)bi * m * k, k);
#else
			for (int i = 0; i < m; i++)
				for (int j = 0; j < k; j++) {
					float s = 0;
					for (int p = 0; p < n; p++)
						s += ((float*)a->data)[bi * m * n + i * n + p] *
						     ((float*)b->data)[p * k + j];
					data[bi * m * k + i * k + j] = s;
				}
#endif
		}
		Tensor* r = make_tensor_arena_f32(data, B * m * k, shape, 3, rg);
		if (rg) tape_append(OP_BMM, r, a, b, 0);
		return r;
	}
	double* data = calloc((size_t)B * m * k, sizeof(double));
	for (int bi = 0; bi < B; bi++) {
#ifdef __APPLE__
		// NOLINTNEXTLINE(misc-include-cleaner): BLAS symbols via Accelerate umbrella
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0,
		            ((double*)a->data) + (size_t)bi * m * n, n, b->data, k, 0.0,
		            data + (size_t)bi * m * k, k);
#else
		for (int i = 0; i < m; i++)
			for (int j = 0; j < k; j++) {
				double s = 0;
				for (int p = 0; p < n; p++)
					s += ((double*)a->data)[bi * m * n + i * n + p] * ((double*)b->data)[p * k + j];
				data[bi * m * k + i * k + j] = s;
			}
#endif
	}
	Tensor* r = make_tensor(data, shape, 3, rg);
	free(data);
	if (rg) tape_append(OP_BMM, r, a, b, 0);
	return r;
}

static void tape_backward_bmm(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	Tensor* b = e->arg2;
	int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2], kk = b->shape[1];
	int is_f32 = (a->dtype_tag == DT_F32);
	ensure_grad(r);
	if (a && a->requires_grad) {
		ensure_grad(a);
		if (is_f32) {
			for (int bi = 0; bi < BB; bi++)
				for (int i = 0; i < mm; i++)
					for (int j = 0; j < nn; j++) {
						double s = 0;
						for (int p = 0; p < kk; p++)
							s += tape_grad_load_d(r, bi * mm * kk + i * kk + p) *
							     tape_load_d(b, j * kk + p);
						tape_grad_add_d(a, bi * mm * nn + i * nn + j, s);
					}
		} else
#ifdef __APPLE__
			// NOLINTNEXTLINE(misc-include-cleaner): BLAS symbols via Accelerate umbrella
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, BB * mm, nn, kk, 1.0, r->grad, kk,
			            b->data, kk, 1.0, a->grad, nn);
#else
			for (int bi = 0; bi < BB; bi++)
				for (int i = 0; i < mm; i++)
					for (int j = 0; j < nn; j++) {
						double s = 0;
						for (int p = 0; p < kk; p++)
							s += tape_grad_load_d(r, bi * mm * kk + i * kk + p) *
							     ((double*)b->data)[j * kk + p];
						tape_grad_add_d(a, bi * mm * nn + i * nn + j, s);
					}
#endif
	}
	if (b && b->requires_grad) {
		ensure_grad(b);
		if (is_f32) {
			for (int bi = 0; bi < BB; bi++)
				for (int j = 0; j < nn; j++)
					for (int p = 0; p < kk; p++) {
						double s = 0;
						for (int i = 0; i < mm; i++)
							s += tape_load_d(a, bi * mm * nn + i * nn + j) *
							     tape_grad_load_d(r, bi * mm * kk + i * kk + p);
						tape_grad_add_d(b, j * kk + p, s);
					}
		} else
#ifdef __APPLE__
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nn, kk, BB * mm, 1.0, a->data, nn,
			            r->grad, kk, 1.0, b->grad, kk);
#else
			for (int bi = 0; bi < BB; bi++)
				for (int j = 0; j < nn; j++)
					for (int p = 0; p < kk; p++) {
						double s = 0;
						for (int i = 0; i < mm; i++)
							s += ((double*)a->data)[bi * mm * nn + i * nn + j] *
							     tape_grad_load_d(r, bi * mm * kk + i * kk + p);
						tape_grad_add_d(b, j * kk + p, s);
					}
#endif
		;
	}
}

TAPE_REGISTER_OP(OP_BMM, tape_backward_bmm)

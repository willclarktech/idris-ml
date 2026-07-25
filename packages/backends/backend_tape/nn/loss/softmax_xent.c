/* nn/loss/softmax_xent.c — fused softmax cross-entropy with logits
 * (forward + backward, one tape node).
 *
 * Formula (soft/one-hot targets, PyTorch F.cross_entropy-with-probs
 * analogue up to the caller-chosen scale):
 *   ls[i,j] = log_softmax(input, rows)[i,j]
 *   out     = -scale * sum_ij target[i,j] * ls[i,j]
 *
 * Replaces the decomposed `log_softmax_2d -> mul -> sum -> neg ->
 * mul_scalar` chain (one tape node instead of five). The forward
 * mirrors that chain's arithmetic EXACTLY on the F64 path — same
 * per-row log-softmax loop as nn/softmax/log_softmax_2d.c, flat
 * row-major `acc += ls*t` accumulation matching
 * linear/reduction/sum.c, then `(-acc) * scale` (negate before scale,
 * matching neg -> mul_scalar) — so adopting the fused op is
 * bit-identical on tape F64 (the backends-README regression bar).
 *
 * Backward (derivation: d ls[i,l]/d x[i,j] = delta(l,j) - p[i,j] with
 * p = softmax = exp(ls); S_i = sum_l target[i,l]):
 *   d input[i,j]  = upstream * scale * (p[i,j] * S_i - target[i,j])
 *   d target[i,j] = upstream * (-scale) * ls[i,j]
 *
 * Rank-1 input is accepted as [1, n]. F32 + F64 paths; the ls cache is
 * always double* so backward reads uniformly.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_softmax_xent_2d(TensorHandle hinput, TensorHandle htarget, double scale) {
	Tensor* t = (Tensor*)hinput;
	Tensor* target = (Tensor*)htarget;
	if (t->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_softmax_xent_2d");
	int m = (t->rank == 1) ? 1 : t->shape[0];
	int n = (t->rank == 1) ? t->shape[0] : t->shape[1];
	int rg = t->requires_grad || target->requires_grad;

	/* calloc, not malloc: the fill loop below writes every element, but it
	 * is bounded by `m` while the accumulation loop is bounded by `m * n`,
	 * and clang's analyzer cannot relate the two — it posits m <= 0 with
	 * m * n > 0 and reports a garbage read. Zero-initialised memory closes
	 * that path; large blocks come back as fresh zero pages, so the cost is
	 * not a memset. */
	double* ls = calloc((size_t)m * n, sizeof(double));
	for (int i = 0; i < m; i++) {
		double max_val = tape_load_d(t, i * n);
		for (int j = 1; j < n; j++) {
			double v = tape_load_d(t, i * n + j);
			if (v > max_val) max_val = v;
		}
		double sum_exp = 0;
		for (int j = 0; j < n; j++)
			sum_exp += exp(tape_load_d(t, i * n + j) - max_val);
		double log_sum = log(sum_exp) + max_val;
		for (int j = 0; j < n; j++)
			ls[i * n + j] = tape_load_d(t, i * n + j) - log_sum;
	}

	/* Flat row-major accumulation, matching the decomposed chain's
	 * mul -> sum ordering bit-for-bit on F64. The product is a separate
	 * statement so clang's default FP contraction can't fuse it into an
	 * fma (the decomposed chain stores the product tensor, so its sum
	 * adds plain rounded products). */
	double acc = 0;
	for (int i = 0; i < m * n; i++) {
		double prod = ls[i] * tape_load_d(target, i);
		acc += prod;
	}
	double loss = (-acc) * scale;

	Tensor* r = (t->dtype_tag == DT_F32) ? make_scalar_f32(loss, rg) : make_scalar(loss, rg);
	if (rg) {
		SoftmaxXentMeta* meta = arena_alloc(sizeof(SoftmaxXentMeta));
		meta->target = target;
		meta->ls = ls;
		meta->m = m;
		meta->n = n;
		meta->scale = scale;
		TapeEntry* e = tape_append(OP_SOFTMAX_XENT_2D, r, t, NULL, 0);
		e->op_meta = meta;
	} else {
		free(ls);
	}
	return r;
}

static void tape_backward_softmax_xent_2d(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	SoftmaxXentMeta* meta = (SoftmaxXentMeta*)e->op_meta;
	int mm = meta->m, nn = meta->n;
	ensure_grad(r);
	/* Replicate the decomposed chain's backward arithmetic BIT-EXACTLY
	 * (mul_scalar -> neg -> sum -> mul -> log_softmax_2d):
	 *   up2       = -(up * scale)          (the grad reaching the ls*t product)
	 *   g_ls[i,j] = up2 * t[i,j]
	 *   per row:  sumg_i = sum_j g_ls[i,j]  (j order)
	 *   g_in[i,j] = g_ls[i,j] - exp(ls[i,j]) * sumg_i
	 * Algebraically up*scale*(softmax*rowsum(t) - t), but the operation
	 * order matches log_softmax_2d.c's backward so switching tnllLossMean
	 * to the fused op leaves full training runs bit-identical on F64. */
	double up2 = -(tape_grad_load_d(r, 0) * meta->scale);
	if (a && a->requires_grad) {
		ensure_grad(a);
		for (int i = 0; i < mm; i++) {
			double sumg = 0;
			for (int j = 0; j < nn; j++) {
				/* Separate statement: the decomposed chain sums STORED
				 * g_ls values, so no fma contraction is allowed here. */
				double gls = up2 * tape_load_d(meta->target, i * nn + j);
				sumg += gls;
			}
			for (int j = 0; j < nn; j++) {
				double gls = up2 * tape_load_d(meta->target, i * nn + j);
				tape_grad_add_d(a, i * nn + j, gls - exp(meta->ls[i * nn + j]) * sumg);
			}
		}
	}
	if (meta->target->requires_grad) {
		ensure_grad(meta->target);
		for (int i = 0; i < mm * nn; i++)
			tape_grad_add_d(meta->target, i, up2 * meta->ls[i]);
	}
}

TAPE_REGISTER_OP(OP_SOFTMAX_XENT_2D, tape_backward_softmax_xent_2d)

/* nn/norm/batch_norm.c — per-channel batch norm (forward + backward).
 *
 * F64 BIT-EXACT RISK — running mean/var accumulation
 * ordering preserved. Training mode updates running stats in-place
 * via tape_store_d. Eval mode reads running stats only.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               TensorHandle hrunning_mean, TensorHandle hrunning_var, int C,
                               int spatial, int training, double momentum, double eps) {
	Tensor* input = (Tensor*)hinput;
	Tensor* gamma = (Tensor*)hgamma;
	Tensor* beta = (Tensor*)hbeta;
	Tensor* running_mean = (Tensor*)hrunning_mean;
	Tensor* running_var = (Tensor*)hrunning_var;
	if (input->dtype_tag != gamma->dtype_tag || input->dtype_tag != beta->dtype_tag ||
	    input->dtype_tag != running_mean->dtype_tag || input->dtype_tag != running_var->dtype_tag)
		tape_abort_mixed_dtype("tensor_batch_norm");
	int n = C * spatial;
	int out_shape[1] = {n};
	int rg = input->requires_grad || gamma->requires_grad || beta->requires_grad;
	int is_f32 = (input->dtype_tag == DT_F32);

	void* out;
	if (is_f32)
		out = arena_alloc(n * sizeof(float));
	else
		out = calloc(n, sizeof(double));
	double* x_hat = malloc(n * sizeof(double));
	double* rstd = malloc(C * sizeof(double));

	for (int c = 0; c < C; c++) {
		double mean, var;
		if (training) {
			mean = 0;
			for (int j = 0; j < spatial; j++)
				mean += tape_load_d(input, c * spatial + j);
			mean /= spatial;
			var = 0;
			for (int j = 0; j < spatial; j++) {
				double d = tape_load_d(input, c * spatial + j) - mean;
				var += d * d;
			}
			var /= spatial;
			double rm = tape_load_d(running_mean, c);
			double rv = tape_load_d(running_var, c);
			/* Bessel n/(n-1) correction on the running-var update only (matches
			 * torch::batch_norm / PyTorch); `var` itself stays biased for the
			 * per-batch normalization below. At spatial==1 this is var*inf=NaN,
			 * matching PyTorch's documented batchnorm-on-one-element behaviour. */
			double bessel = (double)spatial / (spatial - 1.0);
			tape_store_d(running_mean, c, (1.0 - momentum) * rm + momentum * mean);
			tape_store_d(running_var, c, (1.0 - momentum) * rv + momentum * var * bessel);
		} else {
			mean = tape_load_d(running_mean, c);
			var = tape_load_d(running_var, c);
		}

		double rs = 1.0 / sqrt(var + eps);
		rstd[c] = rs;
		double gc = tape_load_d(gamma, c);
		double bc = tape_load_d(beta, c);
		for (int j = 0; j < spatial; j++) {
			int idx = c * spatial + j;
			double xh = (tape_load_d(input, idx) - mean) * rs;
			x_hat[idx] = xh;
			double v = gc * xh + bc;
			if (is_f32)
				((float*)out)[idx] = (float)v;
			else
				((double*)out)[idx] = v;
		}
	}

	Tensor* r;
	if (is_f32) {
		r = make_tensor_arena_f32((float*)out, n, out_shape, 1, rg);
	} else {
		r = make_tensor((double*)out, out_shape, 1, rg);
		free(out);
	}

	if (r->requires_grad) {
		TapeEntry* e = tape_append(OP_BATCH_NORM, r, input, NULL, 0);
		BatchNormMeta* meta = arena_alloc(sizeof(BatchNormMeta));
		meta->gamma = gamma;
		meta->beta = beta;
		meta->x_hat = x_hat;
		meta->rstd = rstd;
		meta->C = C;
		meta->spatial = spatial;
		e->op_meta = meta;
	} else {
		free(x_hat);
		free(rstd);
	}
	return r;
}

static void tape_backward_batch_norm(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	BatchNormMeta* meta = (BatchNormMeta*)e->op_meta;
	int CC = meta->C, sp = meta->spatial;
	ensure_grad(r);
	if (meta->gamma->requires_grad) {
		ensure_grad(meta->gamma);
		for (int c = 0; c < CC; c++) {
			double dg = 0;
			for (int j = 0; j < sp; j++)
				dg += tape_grad_load_d(r, c * sp + j) * meta->x_hat[c * sp + j];
			tape_grad_add_d(meta->gamma, c, dg);
		}
	}
	if (meta->beta->requires_grad) {
		ensure_grad(meta->beta);
		for (int c = 0; c < CC; c++) {
			double db = 0;
			for (int j = 0; j < sp; j++)
				db += tape_grad_load_d(r, c * sp + j);
			tape_grad_add_d(meta->beta, c, db);
		}
	}
	if (a && a->requires_grad) {
		ensure_grad(a);
		for (int c = 0; c < CC; c++) {
			double gc = tape_load_d(meta->gamma, c);
			double mean_dxhat = 0, mean_dxhat_xhat = 0;
			for (int j = 0; j < sp; j++) {
				double dxh = tape_grad_load_d(r, c * sp + j) * gc;
				mean_dxhat += dxh;
				mean_dxhat_xhat += dxh * meta->x_hat[c * sp + j];
			}
			mean_dxhat /= sp;
			mean_dxhat_xhat /= sp;
			for (int j = 0; j < sp; j++) {
				double dxh = tape_grad_load_d(r, c * sp + j) * gc;
				tape_grad_add_d(a, c * sp + j,
				                meta->rstd[c] *
				                    (dxh - mean_dxhat - meta->x_hat[c * sp + j] * mean_dxhat_xhat));
			}
		}
	}
}

TAPE_REGISTER_OP(OP_BATCH_NORM, tape_backward_batch_norm)

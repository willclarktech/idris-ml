/* nn/norm/layer_norm_2d.c — row-wise layer norm (forward + backward).
 *
 * F64 BIT-EXACT RISK — accumulation order preserved
 * verbatim. F32 + F64 paths. x_hat + rstd cached in LayerNormMeta
 * (always double*) so backward reads uniformly.
 */

#include <math.h>
#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_layer_norm_2d(TensorHandle h, TensorHandle hgamma, TensorHandle hbias,
                                  double eps) {
	Tensor* t = (Tensor*)h;
	Tensor* gamma = (Tensor*)hgamma;
	Tensor* bias = (Tensor*)hbias;
	if (t->dtype_tag != gamma->dtype_tag || t->dtype_tag != bias->dtype_tag)
		tape_abort_mixed_dtype("tensor_layer_norm_2d");
	int m = t->shape[0], n = t->shape[1];
	int shape[] = {m, n};
	int rg = t->requires_grad || gamma->requires_grad || bias->requires_grad;

	if (t->dtype_tag == DT_F32) {
		float* data = arena_alloc(m * n * sizeof(float));
		double* x_hat = malloc(m * n * sizeof(double));
		double* rstd = malloc(m * sizeof(double));
		const float* td = (const float*)t->data;
		const float* gd = (const float*)gamma->data;
		const float* bd = (const float*)bias->data;
		for (int i = 0; i < m; i++) {
			double mean = 0;
			for (int j = 0; j < n; j++)
				mean += td[i * n + j];
			mean /= n;
			double var = 0;
			for (int j = 0; j < n; j++) {
				double d = td[i * n + j] - mean;
				var += d * d;
			}
			var /= n;
			double inv_std = 1.0 / sqrt(var + eps);
			rstd[i] = inv_std;
			for (int j = 0; j < n; j++) {
				double xh = (td[i * n + j] - mean) * inv_std;
				x_hat[i * n + j] = xh;
				data[i * n + j] = (float)(gd[j] * xh + bd[j]);
			}
		}
		Tensor* r = make_tensor_arena_f32(data, m * n, shape, 2, rg);
		if (rg) {
			LayerNormMeta* meta = arena_alloc(sizeof(LayerNormMeta));
			meta->gamma = gamma;
			meta->bias = bias;
			meta->x_hat = x_hat;
			meta->rstd = rstd;
			meta->m = m;
			meta->n = n;
			TapeEntry* e = tape_append(OP_LAYER_NORM_2D, r, t, NULL, 0);
			e->op_meta = meta;
		} else {
			free(x_hat);
			free(rstd);
		}
		return r;
	}

	double* data = malloc(m * n * sizeof(double));
	double* x_hat = malloc(m * n * sizeof(double));
	double* rstd = malloc(m * sizeof(double));
	for (int i = 0; i < m; i++) {
		double mean = 0;
		for (int j = 0; j < n; j++)
			mean += ((double*)t->data)[i * n + j];
		mean /= n;
		double var = 0;
		for (int j = 0; j < n; j++) {
			double d = ((double*)t->data)[i * n + j] - mean;
			var += d * d;
		}
		var /= n;
		double inv_std = 1.0 / sqrt(var + eps);
		rstd[i] = inv_std;
		for (int j = 0; j < n; j++) {
			x_hat[i * n + j] = (((double*)t->data)[i * n + j] - mean) * inv_std;
			data[i * n + j] =
			    ((double*)gamma->data)[j] * x_hat[i * n + j] + ((double*)bias->data)[j];
		}
	}
	Tensor* r = make_tensor(data, shape, 2, rg);
	free(data);
	if (rg) {
		LayerNormMeta* meta = arena_alloc(sizeof(LayerNormMeta));
		meta->gamma = gamma;
		meta->bias = bias;
		meta->x_hat = x_hat;
		meta->rstd = rstd;
		meta->m = m;
		meta->n = n;
		TapeEntry* e = tape_append(OP_LAYER_NORM_2D, r, t, NULL, 0);
		e->op_meta = meta;
	} else {
		free(x_hat);
		free(rstd);
	}
	return r;
}

static void tape_backward_layer_norm_2d(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	LayerNormMeta* meta = (LayerNormMeta*)e->op_meta;
	int mm = meta->m, nn = meta->n;
	ensure_grad(r);
	if (meta->gamma && meta->gamma->requires_grad) {
		ensure_grad(meta->gamma);
		for (int j = 0; j < nn; j++) {
			double dg = 0;
			for (int i = 0; i < mm; i++)
				dg += tape_grad_load_d(r, i * nn + j) * meta->x_hat[i * nn + j];
			tape_grad_add_d(meta->gamma, j, dg);
		}
	}
	if (meta->bias && meta->bias->requires_grad) {
		ensure_grad(meta->bias);
		for (int j = 0; j < nn; j++) {
			double db = 0;
			for (int i = 0; i < mm; i++)
				db += tape_grad_load_d(r, i * nn + j);
			tape_grad_add_d(meta->bias, j, db);
		}
	}
	if (a && a->requires_grad) {
		ensure_grad(a);
		for (int i = 0; i < mm; i++) {
			double mean_dxhat = 0;
			double mean_dxhat_xhat = 0;
			for (int j = 0; j < nn; j++) {
				double dxh = tape_grad_load_d(r, i * nn + j) * tape_load_d(meta->gamma, j);
				mean_dxhat += dxh;
				mean_dxhat_xhat += dxh * meta->x_hat[i * nn + j];
			}
			mean_dxhat /= nn;
			mean_dxhat_xhat /= nn;
			for (int j = 0; j < nn; j++) {
				double dxh = tape_grad_load_d(r, i * nn + j) * tape_load_d(meta->gamma, j);
				tape_grad_add_d(a, i * nn + j,
				                meta->rstd[i] *
				                    (dxh - mean_dxhat - meta->x_hat[i * nn + j] * mean_dxhat_xhat));
			}
		}
	}
}

TAPE_REGISTER_OP(OP_LAYER_NORM_2D, tape_backward_layer_norm_2d)

/* training/param_create.c — persistent tensor creation for params + state.
 *
 * tensor_create_param_{1,2,3,4}d and tensor_create_state_{1,2}d.
 * Param tensors are persistent (survive arena resets) with requires_grad=1;
 * state tensors are persistent with requires_grad=0 (used for non-learnable
 * NTM/DNC memory and RNN hidden states).
 *
 * The data buffer passed in is owned by the caller (typically via
 * tensor_alloc_doubles) and freed here after memcpy into the heap-malloc'd
 * tensor body. All paths emit an OP_CONST tape entry for the param so
 * the gradient registry knows about it; state tensors skip the tape
 * entry (they're not gradient-bearing).
 */

#include <stdlib.h>
#include <string.h>
#include "../tape.h"
#include "../tensor.h"
#include "../../backend.h"

TensorHandle tensor_create_param_1d_f64(int n, double* data) {
	Tensor* t = calloc(1, sizeof(Tensor));
	t->data = malloc(n * sizeof(double));
	memcpy(t->data, data, n * sizeof(double));
	free(data);
	t->shape = malloc(sizeof(int));
	t->shape[0] = n;
	t->rank = 1;
	t->numel = n;
	t->requires_grad = 1;
	t->tape_idx = -1;
	t->persistent = 1;
	tape_append(OP_CONST, t, NULL, NULL, 0);
	return t;
}

TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* data) {
	int numel = rows * cols;
	Tensor* t = calloc(1, sizeof(Tensor));
	t->data = malloc(numel * sizeof(double));
	memcpy(t->data, data, numel * sizeof(double));
	free(data);
	t->shape = malloc(2 * sizeof(int));
	t->shape[0] = rows;
	t->shape[1] = cols;
	t->rank = 2;
	t->numel = numel;
	t->requires_grad = 1;
	t->tape_idx = -1;
	t->persistent = 1;
	tape_append(OP_CONST, t, NULL, NULL, 0);
	return t;
}

TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* data) {
	int numel = d0 * d1 * d2;
	Tensor* t = calloc(1, sizeof(Tensor));
	t->data = malloc(numel * sizeof(double));
	memcpy(t->data, data, numel * sizeof(double));
	free(data);
	t->shape = malloc(3 * sizeof(int));
	t->shape[0] = d0;
	t->shape[1] = d1;
	t->shape[2] = d2;
	t->rank = 3;
	t->numel = numel;
	t->requires_grad = 1;
	t->tape_idx = -1;
	t->persistent = 1;
	tape_append(OP_CONST, t, NULL, NULL, 0);
	return t;
}

TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* data) {
	int numel = d0 * d1 * d2 * d3;
	Tensor* t = calloc(1, sizeof(Tensor));
	t->data = malloc(numel * sizeof(double));
	memcpy(t->data, data, numel * sizeof(double));
	free(data);
	t->shape = malloc(4 * sizeof(int));
	t->shape[0] = d0;
	t->shape[1] = d1;
	t->shape[2] = d2;
	t->shape[3] = d3;
	t->rank = 4;
	t->numel = numel;
	t->requires_grad = 1;
	t->tape_idx = -1;
	t->persistent = 1;
	tape_append(OP_CONST, t, NULL, NULL, 0);
	return t;
}

/* Persistent tensors WITHOUT requires_grad — for non-learnable NTM state */

TensorHandle tensor_create_state_1d_f64(int n, double* data) {
	Tensor* t = calloc(1, sizeof(Tensor));
	t->data = malloc(n * sizeof(double));
	memcpy(t->data, data, n * sizeof(double));
	free(data);
	t->shape = malloc(sizeof(int));
	t->shape[0] = n;
	t->rank = 1;
	t->numel = n;
	t->requires_grad = 0;
	t->tape_idx = -1;
	t->persistent = 1;
	return t;
}

TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* data) {
	int numel = rows * cols;
	Tensor* t = calloc(1, sizeof(Tensor));
	t->data = malloc(numel * sizeof(double));
	memcpy(t->data, data, numel * sizeof(double));
	free(data);
	t->shape = malloc(2 * sizeof(int));
	t->shape[0] = rows;
	t->shape[1] = cols;
	t->rank = 2;
	t->numel = numel;
	t->requires_grad = 0;
	t->tape_idx = -1;
	t->persistent = 1;
	return t;
}

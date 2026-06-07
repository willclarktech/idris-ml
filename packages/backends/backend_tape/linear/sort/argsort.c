/* linear/sort/argsort.c — argsort (forward only — non-differentiable).
 *
 * Sorts indices by tensor values; result has
 * requires_grad=0. F32 and F64 each use a dedicated comparator with
 * a TU-local data pointer.
 */

#include <stdlib.h>
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

static const double* argsort_data_ptr;
static int argsort_cmp_asc(const void* a, const void* b) {
	int ia = *(const int*)a, ib = *(const int*)b;
	double da = argsort_data_ptr[ia], db = argsort_data_ptr[ib];
	return (da > db) - (da < db);
}
static int argsort_cmp_desc(const void* a, const void* b) {
	int ia = *(const int*)a, ib = *(const int*)b;
	double da = argsort_data_ptr[ia], db = argsort_data_ptr[ib];
	return (db > da) - (db < da);
}
static const float* argsort_data_ptr_f32;
static int argsort_cmp_asc_f32(const void* a, const void* b) {
	int ia = *(const int*)a, ib = *(const int*)b;
	float da = argsort_data_ptr_f32[ia], db = argsort_data_ptr_f32[ib];
	return (da > db) - (da < db);
}
static int argsort_cmp_desc_f32(const void* a, const void* b) {
	int ia = *(const int*)a, ib = *(const int*)b;
	float da = argsort_data_ptr_f32[ia], db = argsort_data_ptr_f32[ib];
	return (db > da) - (db < da);
}

TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
	(void)dim;
	Tensor* t = (Tensor*)ht;
	int n = t->numel;
	int* indices = malloc(n * sizeof(int));
	for (int i = 0; i < n; i++)
		indices[i] = i;
	if (t->dtype_tag == DT_F32) {
		argsort_data_ptr_f32 = (const float*)t->data;
		qsort(indices, n, sizeof(int), descending ? argsort_cmp_desc_f32 : argsort_cmp_asc_f32);
	} else {
		argsort_data_ptr = t->data;
		qsort(indices, n, sizeof(int), descending ? argsort_cmp_desc : argsort_cmp_asc);
	}
	int shape[] = {n};
	Tensor* r;
	if (t->dtype_tag == DT_F32) {
		float* out = arena_alloc(n * sizeof(float));
		for (int i = 0; i < n; i++)
			out[i] = (float)indices[i];
		r = make_tensor_arena_f32(out, n, shape, 1, 0);
	} else {
		double* out = malloc(n * sizeof(double));
		for (int i = 0; i < n; i++)
			out[i] = (double)indices[i];
		r = make_tensor(out, shape, 1, 0);
		free(out);
	}
	free(indices);
	return r;
}

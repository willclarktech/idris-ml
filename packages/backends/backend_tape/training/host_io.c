/* training/host_io.c — host-side tensor inspection.
 *
 * Shape queries (tensor_numel/dim/size), host readout
 * (tensor_to_doubles/floats/int64), and the dtype-name accessor.
 *
 * tensor_to_int64 follows the byte-level I64 readout contract from
 * backend.h: tape has no native int storage (integer dtypes ride
 * in double* via the lingua-franca path, rounded on store), so this
 * casts from the dtype-uniform double view. Matches the value the
 * safetensors double path was producing pre-row-20; no regression.
 */

#include <stdint.h>
#include <string.h>
#include "../arena.h"
#include "../tensor.h"
#include "../../backend.h"

int tensor_numel(TensorHandle h) {
	return ((Tensor*)h)->numel;
}
int tensor_dim(TensorHandle h) {
	return ((Tensor*)h)->rank;
}

int tensor_size(TensorHandle h, int dim) {
	Tensor* t = (Tensor*)h;
	if (dim < t->rank) return t->shape[dim];
	return 0;
}

void tensor_to_doubles(TensorHandle h, double* out) {
	Tensor* t = (Tensor*)h;
	if (t->dtype_tag == DT_F32) {
		for (int i = 0; i < t->numel; i++)
			out[i] = (double)((float*)t->data)[i];
	} else {
		memcpy(out, t->data, t->numel * sizeof(double));
	}
}

void tensor_to_int64(TensorHandle h, int64_t* out) {
	Tensor* t = (Tensor*)h;
	for (int i = 0; i < t->numel; i++) {
		out[i] = (int64_t)tape_load_d(t, i);
	}
}

void tensor_to_floats(TensorHandle h, float* out) {
	Tensor* t = (Tensor*)h;
	if (t->dtype_tag == DT_F32) {
		memcpy(out, t->data, t->numel * sizeof(float));
	} else {
		for (int i = 0; i < t->numel; i++)
			out[i] = (float)((double*)t->data)[i];
	}
}

const char* tensor_dtype_name(TensorHandle h) {
	switch (((Tensor*)h)->dtype_tag) {
	case DT_F32:
		return "F32";
	case DT_BF16:
		return "BF16";
	case DT_F16:
		return "F16";
	case DT_I8:
		return "I8";
	case DT_I16:
		return "I16";
	case DT_I32:
		return "I32";
	case DT_I64:
		return "I64";
	case DT_U8:
		return "U8";
	case DT_BOOL:
		return "BOOL";
	default:
		return "F64"; /* DT_F64 */
	}
}

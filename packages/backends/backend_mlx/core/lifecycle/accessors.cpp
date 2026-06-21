/* Tensor accessors — mlx.
 *
 *   - tensor_numel / tensor_dim / tensor_size: shape introspection.
 *   - tensor_to_doubles / tensor_to_floats: host readback bridges.
 *     `tensor_to_floats` fast-paths F32 (memcpy-style loop over the
 *     native buffer); BF16 widens per-element via the bfloat16_t scalar
 *     conversion; F64 goes through the lingua-franca cast.
 *   - tensor_to_int64: byte-level I64 readout. mlx has no native int64
 *     storage; integer round-trip goes through double, so this inherits
 *     the 2^53 ceiling. Implemented for symbol completeness — the
 *     Idris-side `Compatible MlxExecutor I64` is closed, so the realistic
 *     reachable use is safetensors I/O on F32/F64/BF16-typed tensors.
 *   - tensor_dtype_name: F32, F64, BF16, F16, or I32. Other int* +
 *     bool are not wired (Idris `Compatible` gates each pair at the
 *     type level).
 *
 * `tensor_item` (scalar readout) lives in core/lifecycle/item.cpp.
 */
#include "../../tensor.h"
#include "../../precision.h"
#include <cstdlib>
#include <cstdint>

extern "C" int tensor_numel(TensorHandle h) {
	return (int)((Tensor*)h)->data.size();
}
extern "C" int tensor_dim(TensorHandle h) {
	return (int)((Tensor*)h)->data.ndim();
}
extern "C" int tensor_size(TensorHandle h, int dim) {
	return (int)((Tensor*)h)->data.shape(dim);
}

extern "C" void tensor_to_doubles(TensorHandle h, double* out) {
	auto* t = (Tensor*)h;
	/* Force contiguous storage first — mlx ops like transpose return
	 * strided views over the original buffer; reading via data<T>()
	 * would walk storage order, not logical order. mx::contiguous
	 * materializes a fresh contiguous copy. Same pattern the
	 * adapter/optimizer/diagnostics readouts use (search
	 * "mx::contiguous" under backend_mlx/training/). Without this,
	 * any transposed/reshaped/narrowed view read by a user crashes
	 * with corrupted values — see W3 OP_TRANSPOSE_LAST2 test. */
	auto contig = mx::contiguous(t->data);
	mx::eval(contig);
	mx_to_doubles(contig, out);
}

// Byte-level I64 readout — declared in backend.h with the byte-exact
// contract honoured only on backends with native int64 storage. mlx
// stores only F32/F64; integer storage round-trips through `double`,
// inheriting the same 2^53 ceiling as the lingua-franca path.
// Practically the safetensors I/O caller only reaches this on tensors
// already typed I64, which mlx can't construct (Compatible MlxExecutor I64
// is closed). Implemented for symbol completeness.
extern "C" void tensor_to_int64(TensorHandle h, int64_t* out) {
	auto* t = (Tensor*)h;
	/* See tensor_to_doubles re: mx::contiguous. Same requirement here. */
	auto contig = mx::contiguous(t->data);
	mx::eval(contig);
	int const n = (int)contig.size();
	double* tmp = (double*)malloc((size_t)n * sizeof(double));
	if (tmp == nullptr) return;
	mx_to_doubles(contig, tmp);
	for (int i = 0; i < n; i++)
		out[i] = (int64_t)tmp[i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
	free(tmp);
}

extern "C" void tensor_to_floats(TensorHandle h, float* out) {
	auto* t = (Tensor*)h;
	/* See tensor_to_doubles re: mx::contiguous. Same requirement here. */
	auto contig = mx::contiguous(t->data);
	mx::eval(contig);
	int const n = (int)contig.size();
	if (contig.dtype() == mx::float32) {
		const float* src = contig.data<float>();
		for (int i = 0; i < n; i++)
			out[i] = src[i];
	} else if (contig.dtype() == mx::bfloat16) {
		const mx::bfloat16_t* src = contig.data<mx::bfloat16_t>();
		for (int i = 0; i < n; i++)
			out[i] = (float)src[i];
	} else if (contig.dtype() == mx::float16) {
		const mx::float16_t* src = contig.data<mx::float16_t>();
		for (int i = 0; i < n; i++)
			out[i] = (float)src[i];
	} else {
		const double* src = contig.data<double>();
		for (int i = 0; i < n; i++)
			out[i] = (float)src[i];
	}
}

extern "C" const char* tensor_dtype_name(TensorHandle h) {
	auto* t = (Tensor*)h;
	auto dt = t->data.dtype();
	if (dt == mx::float32) return "F32";
	if (dt == mx::bfloat16) return "BF16";
	if (dt == mx::float16) return "F16";
	if (dt == mx::int32) return "I32";
	return "F64";
}

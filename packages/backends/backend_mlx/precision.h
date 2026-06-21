/* Precision / dtype-bridge helpers for the mlx backend's modular tree.
 *
 * Every op that mixes a Python-typed scalar (eps, momentum, mask value,
 * optimizer state, etc.) into an mx::array must match the array's
 * dtype, or the result silently downcasts (an f64 path then computes
 * in f32 with no warning). `scalar_like(v, ref)` builds the scalar in
 * `ref`'s dtype so the math stays at the type-level claim. The named
 * `zero_like` / `one_like` / `half_like` cover the three hot-path
 * constants the legacy code held as f32 singletons.
 *
 * The mx_to_doubles / mx_read_double / mx_from_doubles /
 * mx_array_from_doubles helpers bridge the public C `double*` ABI to
 * mlx's per-storage-type buffers (f32 widens per-element, f64 copies
 * through).
 */
#ifndef IDRISML_BACKEND_MLX_PRECISION_H
#define IDRISML_BACKEND_MLX_PRECISION_H

#include <vector>
#include "tensor.h" /* for `namespace mx = mlx::core;` */

inline mx::array scalar_like(double v, const mx::array& ref) {
	return mx::array(v, ref.dtype());
}

inline mx::array zero_like(const mx::array& ref) {
	return scalar_like(0.0, ref);
}
inline mx::array one_like(const mx::array& ref) {
	return scalar_like(1.0, ref);
}
inline mx::array half_like(const mx::array& ref) {
	return scalar_like(0.5, ref);
}

/* Hot-path f32 singletons. Heap-allocated, intentionally never freed —
   destruction would race against mlx's internal statics on macOS VMs
   (the allocator throws when its backing device is already gone), so
   leaking ~12 bytes + a tiny mlx buffer at exit is the right trade.
   Static-local pointer initialization is thread-safe under C++11+. */
inline const mx::array& kF32_ZERO() {
	static const mx::array* v = new mx::array(0.0f, mx::float32);
	return *v;
}
inline const mx::array& kF32_ONE() {
	static const mx::array* v = new mx::array(1.0f, mx::float32);
	return *v;
}
inline const mx::array& kF32_HALF() {
	static const mx::array* v = new mx::array(0.5f, mx::float32);
	return *v;
}

/* Convert an mlx array to a double buffer (caller-allocated). Branches
   on the array's dtype: f64 sources copy through; f32 widens; bf16/f16
   widen via the mlx scalar conversion. Future dtypes (int*) plug in
   here. */
inline void mx_to_doubles(const mx::array& a, double* out) {
	int const n = (int)a.size();
	if (a.dtype() == mx::float64) {
		const double* src = a.data<double>();
		for (int i = 0; i < n; i++)
			out[i] = src[i];
	} else if (a.dtype() == mx::bfloat16) {
		const mx::bfloat16_t* src = a.data<mx::bfloat16_t>();
		for (int i = 0; i < n; i++)
			out[i] = (double)(float)src[i];
	} else if (a.dtype() == mx::float16) {
		const mx::float16_t* src = a.data<mx::float16_t>();
		for (int i = 0; i < n; i++)
			out[i] = (double)(float)src[i];
	} else if (a.dtype() == mx::int32) {
		const int32_t* src = a.data<int32_t>();
		for (int i = 0; i < n; i++)
			out[i] = (double)src[i];
	} else {
		const float* src = a.data<float>();
		for (int i = 0; i < n; i++)
			out[i] = (double)src[i];
	}
}

/* Read a single element from an mlx array as a double, dtype-aware. */
inline double mx_read_double(const mx::array& a, long idx) {
	if (a.dtype() == mx::float64) return a.data<double>()[idx];
	if (a.dtype() == mx::bfloat16) return (double)(float)a.data<mx::bfloat16_t>()[idx];
	if (a.dtype() == mx::float16) return (double)(float)a.data<mx::float16_t>()[idx];
	if (a.dtype() == mx::int32) return (double)a.data<int32_t>()[idx];
	return (double)a.data<float>()[idx];
}

/* Construct a float32 mx::array from a double buffer + shape. */
inline mx::array mx_from_doubles(const double* data, const mx::Shape& shape) {
	int n = 1;
	for (auto s : shape)
		n *= (int)s;
	std::vector<float> tmp((size_t)n);
	for (int i = 0; i < n; i++)
		tmp[i] = (float)data[i];
	return mx::array(tmp.data(), shape, mx::float32);
}

/* Construct a bfloat16 mx::array from a double buffer + shape.
   Goes via mlx's astype so the float→bfloat16 narrowing uses the
   correct round-to-nearest-even encoding (rather than a hand-rolled
   bit-fiddle). */
inline mx::array mx_bf16_from_doubles(const double* data, const mx::Shape& shape) {
	int n = 1;
	for (auto s : shape)
		n *= (int)s;
	std::vector<float> tmp((size_t)n);
	for (int i = 0; i < n; i++)
		tmp[i] = (float)data[i];
	auto fp32 = mx::array(tmp.data(), shape, mx::float32);
	return mx::astype(fp32, mx::bfloat16);
}

/* Construct a float16 mx::array from a double buffer + shape. Same
   F32-staged narrowing pattern as bf16. */
inline mx::array mx_f16_from_doubles(const double* data, const mx::Shape& shape) {
	int n = 1;
	for (auto s : shape)
		n *= (int)s;
	std::vector<float> tmp((size_t)n);
	for (int i = 0; i < n; i++)
		tmp[i] = (float)data[i];
	auto fp32 = mx::array(tmp.data(), shape, mx::float32);
	return mx::astype(fp32, mx::float16);
}

/* Construct an int32 mx::array from a double buffer + shape. Truncates
   each double to int32_t; out-of-range values follow C cast semantics
   (implementation-defined for >INT_MAX inputs, but mlx never feeds
   those — callers are expected to pass values already representable
   as int32). */
inline mx::array mx_i32_from_doubles(const double* data, const mx::Shape& shape) {
	int n = 1;
	for (auto s : shape)
		n *= (int)s;
	std::vector<int32_t> tmp((size_t)n);
	for (int i = 0; i < n; i++)
		tmp[i] = (int32_t)data[i];
	return mx::array(tmp.data(), shape, mx::int32);
}

/* Construct an mx::array of the requested dtype from a double buffer.
   For float64 storage, pass the buffer through unchanged (lossless).
   For float32 storage, convert per-element (lossy at allocation).
   For bfloat16/float16 storage, widen to F32 then narrow via astype.
   For int32 storage, truncate each double to int32_t. */
inline mx::array mx_array_from_doubles(const double* data, const mx::Shape& shape, mx::Dtype dt) {
	if (dt == mx::float64) {
		return mx::array(data, shape, mx::float64);
	}
	if (dt == mx::bfloat16) {
		return mx_bf16_from_doubles(data, shape);
	}
	if (dt == mx::float16) {
		return mx_f16_from_doubles(data, shape);
	}
	if (dt == mx::int32) {
		return mx_i32_from_doubles(data, shape);
	}
	return mx_from_doubles(data, shape);
}

#endif /* IDRISML_BACKEND_MLX_PRECISION_H */

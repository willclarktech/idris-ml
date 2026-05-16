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
#include "tensor.h"   /* for `namespace mx = mlx::core;` */

inline mx::array scalar_like(double v, const mx::array& ref) {
    return mx::array(v, ref.dtype());
}

inline mx::array zero_like(const mx::array& ref) { return scalar_like(0.0, ref); }
inline mx::array one_like(const mx::array& ref)  { return scalar_like(1.0, ref); }
inline mx::array half_like(const mx::array& ref) { return scalar_like(0.5, ref); }

/* Convert an mlx array to a double buffer (caller-allocated). Branches
   on the array's dtype: f32 sources widen per-element; f64 sources
   copy through. Future dtypes (bf16/fp16/int*) plug in here. */
inline void mx_to_doubles(const mx::array& a, double* out) {
    int n = (int)a.size();
    if (a.dtype() == mx::float64) {
        const double* src = a.data<double>();
        for (int i = 0; i < n; i++) out[i] = src[i];
    } else {
        const float* src = a.data<float>();
        for (int i = 0; i < n; i++) out[i] = (double)src[i];
    }
}

/* Read a single element from an mlx array as a double, dtype-aware. */
inline double mx_read_double(const mx::array& a, long idx) {
    if (a.dtype() == mx::float64) return a.data<double>()[idx];
    return (double)a.data<float>()[idx];
}

/* Construct a float32 mx::array from a double buffer + shape. */
inline mx::array mx_from_doubles(const double* data,
                                 const mx::Shape& shape) {
    int n = 1;
    for (auto s : shape) n *= (int)s;
    std::vector<float> tmp((size_t)n);
    for (int i = 0; i < n; i++) tmp[i] = (float)data[i];
    return mx::array(tmp.data(), shape, mx::float32);
}

/* Construct an mx::array of the requested dtype from a double buffer.
   For float64 storage, pass the buffer through unchanged (lossless).
   For float32 storage, convert per-element (lossy at allocation).
   Future dtypes (bf16, fp16, int*) plug in here. */
inline mx::array mx_array_from_doubles(const double* data,
                                       const mx::Shape& shape,
                                       mx::Dtype dt) {
    if (dt == mx::float64) {
        return mx::array(data, shape, mx::float64);
    }
    return mx_from_doubles(data, shape);
}

#endif /* IDRISML_BACKEND_MLX_PRECISION_H */

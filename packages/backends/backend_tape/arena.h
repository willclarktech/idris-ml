/* backend_tape/arena.h — bump-pointer arena + make_tensor variants +
 * ensure_grad + dtype-aware element load/store.
 *
 * Phase 1.0.4: standalone header (multi-TU compile). Function
 * definitions live in arena.c; hot-path single-line accessors
 * (tape_load_d, tape_store_d) are `static inline` here so the
 * compiler can inline them at every call site without LTO.
 */

#ifndef IDRISML_BACKEND_TAPE_ARENA_H
#define IDRISML_BACKEND_TAPE_ARENA_H

#include <stddef.h>
#include "tensor.h"

/* Arena lifecycle */
void* arena_alloc(size_t bytes);
void  arena_reset(void);

/* make_*: arena-allocated Tensor constructors */
Tensor* make_scalar(double val, int requires_grad);
Tensor* make_tensor(double* data, int* shape, int rank, int requires_grad);
Tensor* make_tensor_arena(double* arena_data, int numel, int* shape, int rank, int requires_grad);
Tensor* make_scalar_f32(double val, int requires_grad);
Tensor* make_tensor_arena_f32(float* arena_data, int numel, int* shape, int rank, int requires_grad);

/* SFX(...) aliases for the .inc machinery (kernel.inc resolves SFX(make_scalar)
 * to make_scalar_f64 / make_scalar_f32 via macro substitution). */
static inline Tensor* make_scalar_f64(double val, int rg) { return make_scalar(val, rg); }
static inline Tensor* make_tensor_arena_f64(double* arena_data, int numel, int* shape, int rank, int rg) {
    return make_tensor_arena(arena_data, numel, shape, rank, rg);
}

/* Grad allocator — grads stay F64 regardless of param dtype. */
void ensure_grad(Tensor* t);

/* Dtype-aware element load — returns t->data[i] cast to double.
 * Hot-path; inline at every call site. For F64 (the common case),
 * one double load — same instruction count as ((double*)t->data)[i]. */
static inline double tape_load_d(const Tensor* t, int i) {
    return (t->dtype_tag == DT_F32) ? (double)((float*)t->data)[i]
                                    : ((double*)t->data)[i];
}

/* Dtype-aware element store — narrows to float when t is F32-tagged. */
static inline void tape_store_d(Tensor* t, int i, double v) {
    if (t->dtype_tag == DT_F32) ((float*)t->data)[i] = (float)v;
    else                        ((double*)t->data)[i] = v;
}

#endif /* IDRISML_BACKEND_TAPE_ARENA_H */

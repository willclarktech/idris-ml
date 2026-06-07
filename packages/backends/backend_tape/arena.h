/* backend_tape/arena.h — bump-pointer arena + make_tensor variants +
 * ensure_grad + dtype-aware element load/store.
 *
 * Standalone header (multi-TU compile). Function
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
void arena_reset(void);
/* arena_free_all: walk the chunk list, free(c->data) + free(c) per chunk,
 * null arena_head / arena_current. Invalidates every previously-returned
 * arena pointer; call only from backend_release_all_persistent (end-of-main
 * pre-exit cleanup), never during a live forward. Moves the multi-GB libc
 * teardown inside main where it's bounded + timeable instead of leaking
 * to a 17-min process-exit tail. */
void arena_free_all(void);

/* make_*: arena-allocated Tensor constructors */
Tensor* make_scalar(double val, int requires_grad);
Tensor* make_tensor(double* data, int* shape, int rank, int requires_grad);
Tensor* make_tensor_arena(double* arena_data, int numel, int* shape, int rank, int requires_grad);
Tensor* make_scalar_f32(double val, int requires_grad);
Tensor* make_tensor_arena_f32(float* arena_data, int numel, int* shape, int rank,
                              int requires_grad);

/* tape_zero_tensor: arena-allocated zero tensor of arbitrary shape and dtype.
 * Used by BLAS-backed kernels (mm/mv/linear/linear_2d/bmm/bmm_3x3) as the
 * zero-dim-guard short-circuit return — cblas rejects lda=0, but the
 * mathematical answer is a properly-shaped zero. Skip tape_append on the
 * caller side: a constant-zero result has zero gradient w.r.t. its inputs. */
Tensor* tape_zero_tensor(int* shape, int rank, int dtype_tag, int requires_grad);

/* SFX(...) aliases for the .inc machinery (kernel.inc resolves SFX(make_scalar)
 * to make_scalar_f64 / make_scalar_f32 via macro substitution). */
static inline Tensor* make_scalar_f64(double val, int rg) {
	return make_scalar(val, rg);
}
static inline Tensor* make_tensor_arena_f64(double* arena_data, int numel, int* shape, int rank,
                                            int rg) {
	return make_tensor_arena(arena_data, numel, shape, rank, rg);
}

/* Grad allocator — buffer size matches data dtype. F32-tagged tensors
 * get a `numel * sizeof(float)` buffer; F64 (and BF16/F16 via the F64
 * lingua-franca path) get `numel * sizeof(double)`. Every backward
 * site must access the buffer via the typed `tape_grad_*` accessors
 * below — direct `((double*)t->grad)[i]` access on F32 tensors would
 * read/write 8 bytes into a 4-byte slot (silent memory smash). */
void ensure_grad(Tensor* t);

/* Byte size of one grad element for the given DT_* tag. F32 → 4,
 * everything else → 8 (the F64 lingua-franca default). */
size_t tape_grad_elem_size(int dtype_tag);

/* Dtype-aware element load — returns t->data[i] cast to double.
 * Hot-path; inline at every call site. For F64 (the common case),
 * one double load — same instruction count as ((double*)t->data)[i]. */
static inline double tape_load_d(const Tensor* t, int i) {
	return (t->dtype_tag == DT_F32) ? (double)((float*)t->data)[i] : ((double*)t->data)[i];
}

/* Dtype-aware element store — narrows to float when t is F32-tagged. */
static inline void tape_store_d(Tensor* t, int i, double v) {
	if (t->dtype_tag == DT_F32)
		((float*)t->data)[i] = (float)v;
	else
		((double*)t->data)[i] = v;
}

/* Dtype-aware grad element load — reads t->grad[i] as double. The
 * buffer's element width matches t->dtype_tag (see `ensure_grad`).
 * For F64 (default), a single double load. */
static inline double tape_grad_load_d(const Tensor* t, int i) {
	return (t->dtype_tag == DT_F32) ? (double)((float*)t->grad)[i] : ((double*)t->grad)[i];
}

/* Dtype-aware grad accumulator — t->grad[i] += v, narrowing to float
 * when t is F32-tagged. Hot-path; backward inner-loop callers should
 * compute the contribution in F64 and store-through with this. */
static inline void tape_grad_add_d(Tensor* t, int i, double v) {
	if (t->dtype_tag == DT_F32)
		((float*)t->grad)[i] += (float)v;
	else
		((double*)t->grad)[i] += v;
}

/* Dtype-aware grad element store — overwrites t->grad[i] with v,
 * narrowing to float when t is F32-tagged. */
static inline void tape_grad_store_d(Tensor* t, int i, double v) {
	if (t->dtype_tag == DT_F32)
		((float*)t->grad)[i] = (float)v;
	else
		((double*)t->grad)[i] = v;
}

#endif /* IDRISML_BACKEND_TAPE_ARENA_H */

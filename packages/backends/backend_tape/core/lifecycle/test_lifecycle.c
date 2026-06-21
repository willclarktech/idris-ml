/* Criterion suite for tape core/lifecycle ops.
 *
 * Covers: tensor_create_scalar, tensor_create, tensor_clone,
 *         tensor_free, tensor_item.
 * tensor_retain_handle / tensor_release_handle are tape-side no-ops
 * (ABI parity stubs); a separate smoke probe verifies they don't crash.
 */

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <criterion/criterion.h>
#include "backend.h"
#include "shared_utils.h" /* tensor_ptr_array_alloc / _set_return / _free */

/* Streamed creators are callee-owns (they FREE their data argument), so every
   streamed data pointer is routed through a fresh heap copy. */
static double* hcopy(const double* s, int n) {
	double* b = malloc((size_t)n * sizeof(double));
	memcpy(b, s, (size_t)n * sizeof(double));
	return b;
}

/* Back-compat aliases live in lifecycle_ext.c but are not part of the public
   backend.h surface (no live caller). Declared here so this colocated suite
   can exercise them — the symbols are present in the linked backend object. */
extern TensorHandle tensor_mul_elementwise(TensorHandle a, TensorHandle b);
extern TensorHandle tensor_sum_all(TensorHandle h);

Test(core_lifecycle, create_scalar_then_item) {
	TensorHandle s = tensor_create_scalar(6.0, 0);
	cr_assert_float_eq(tensor_item(s), 6.0, 1e-12,
	                   "tensor_item should round-trip the value passed to tensor_create_scalar");
	cr_assert_eq(tensor_numel(s), 1);
	cr_assert_eq(tensor_dim(s), 0);
}

Test(core_lifecycle, create_scalar_requires_grad_flag) {
	/* requires_grad threads through to the tensor; the scalar with
	   requires_grad=1 should hold its grad slot ready. We don't read
	   a grad here (no backward), just that the surface accepts the flag
	   and the value still round-trips. */
	TensorHandle s_grad = tensor_create_scalar(2.5, 1);
	TensorHandle s_nograd = tensor_create_scalar(2.5, 0);
	cr_assert_float_eq(tensor_item(s_grad), 2.5, 1e-12);
	cr_assert_float_eq(tensor_item(s_nograd), 2.5, 1e-12);
}

Test(core_lifecycle, create_vector) {
	double data[] = {1.0, 2.0, 3.0};
	int shape[] = {3};
	TensorHandle v = tensor_create(data, shape, 1, 0);
	cr_assert_eq(tensor_numel(v), 3);
	cr_assert_eq(tensor_dim(v), 1);
	cr_assert_eq(tensor_size(v, 0), 3);
	double out[3];
	tensor_to_doubles(v, out);
	cr_assert_float_eq(out[0], 1.0, 1e-12);
	cr_assert_float_eq(out[1], 2.0, 1e-12);
	cr_assert_float_eq(out[2], 3.0, 1e-12);
}

Test(core_lifecycle, create_matrix) {
	double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {2, 3};
	TensorHandle m = tensor_create(data, shape, 2, 0);
	cr_assert_eq(tensor_numel(m), 6);
	cr_assert_eq(tensor_dim(m), 2);
	cr_assert_eq(tensor_size(m, 0), 2);
	cr_assert_eq(tensor_size(m, 1), 3);
}

Test(core_lifecycle, clone_scalar) {
	TensorHandle a = tensor_create_scalar(7.0, 0);
	TensorHandle b = tensor_clone(a);
	cr_assert_float_eq(tensor_item(b), 7.0, 1e-12, "clone should preserve the scalar value");
	cr_assert_eq(tensor_numel(b), 1);
	cr_assert_eq(tensor_dim(b), 0);
	/* Distinct handles (different pointers) — clone is a deep copy. */
	cr_assert_neq((void*)a, (void*)b, "clone must be a new handle");
}

Test(core_lifecycle, clone_vector) {
	double data[] = {10.0, 20.0, 30.0};
	int shape[] = {3};
	TensorHandle a = tensor_create(data, shape, 1, 0);
	TensorHandle b = tensor_clone(a);
	double out[3];
	tensor_to_doubles(b, out);
	cr_assert_float_eq(out[0], 10.0, 1e-12);
	cr_assert_float_eq(out[1], 20.0, 1e-12);
	cr_assert_float_eq(out[2], 30.0, 1e-12);
	cr_assert_neq((void*)a, (void*)b);
}

Test(core_lifecycle, free_is_safe_noop) {
	/* tensor_free is a no-op on tape (arena lifecycle owns teardown).
	   Verify it doesn't crash; subsequent use should still work since
	   the tape holds the underlying pointer alive until tape_reset. */
	TensorHandle s = tensor_create_scalar(1.0, 0);
	tensor_free(s);
	/* Calling tensor_item after free is technically UB on backends
	   that DO free; tape leaves it valid. We don't assert read-back
	   (forward-compat) — just that free itself doesn't crash. */
	cr_assert(1);
}

Test(core_lifecycle, retain_release_handle_noop) {
	/* ABI parity stubs — should not crash for any handle, including
	   NULL (mlx-equivalent does refcount and would crash on NULL). */
	TensorHandle s = tensor_create_scalar(42.0, 0);
	tensor_retain_handle(s);
	tensor_release_handle(s);
	/* Verify value still readable after retain/release dance. */
	cr_assert_float_eq(tensor_item(s), 42.0, 1e-12);
}

Test(core_lifecycle, reshape_1d_collapses_rank) {
	/* tensor_reshape_1d: [2,3] -> [6], storage shared, values preserved. */
	double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {2, 3};
	TensorHandle m = tensor_create(data, shape, 2, 0);
	TensorHandle v = tensor_reshape_1d(m, 6);
	cr_assert_eq(tensor_dim(v), 1);
	cr_assert_eq(tensor_size(v, 0), 6);
	cr_assert_eq(tensor_numel(v), 6);
	double out[6];
	tensor_to_doubles(v, out);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], data[i], 1e-12, "reshape_1d cell %d", i);
}

Test(core_lifecycle, one_hot_encodes_tokens) {
	/* tensor_one_hot: tokens [n] -> flat [n * vocab]. The fn FREES the
	   tokens array it is handed, so it must be heap-allocated. One in-range
	   token sets a single 1.0 per row; the rest of the row stays 0. */
	int n_tokens = 3, vocab = 4;
	int* tokens = malloc(n_tokens * sizeof(int));
	tokens[0] = 0;
	tokens[1] = 2;
	tokens[2] = 3;
	TensorHandle oh = tensor_one_hot(tokens, n_tokens, vocab, 0 /* dtag no-op */);
	cr_assert_eq(tensor_dim(oh), 1);
	cr_assert_eq(tensor_numel(oh), n_tokens * vocab);
	double out[12];
	tensor_to_doubles(oh, out);
	double expect[12] = {1, 0, 0, 0, /* tok 0 */
	                     0, 0, 1, 0, /* tok 2 */
	                     0, 0, 0, 1 /* tok 3 */};
	for (int i = 0; i < 12; i++)
		cr_assert_float_eq(out[i], expect[i], 1e-12, "one_hot cell %d", i);
}

Test(core_lifecycle, one_hot_ignores_out_of_range_token) {
	/* The `tok >= 0 && tok < vocab_size` guard: an out-of-range token leaves
	   its row all-zero (no write), exercising the false branch. */
	int n_tokens = 2, vocab = 3;
	int* tokens = malloc(n_tokens * sizeof(int));
	tokens[0] = 5;  /* >= vocab -> skipped */
	tokens[1] = -1; /* < 0      -> skipped */
	TensorHandle oh = tensor_one_hot(tokens, n_tokens, vocab, 0);
	cr_assert_eq(tensor_numel(oh), n_tokens * vocab);
	double out[6];
	tensor_to_doubles(oh, out);
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], 0.0, 1e-12, "out-of-range one_hot cell %d", i);
}

Test(core_lifecycle, subtract_scalar_inplace_mutates_f64) {
	/* tensor_subtract_scalar_inplace: F64 branch subtracts val from every
	   element in place and returns the same handle. */
	double data[] = {10.0, 20.0, 30.0};
	int shape[] = {3};
	TensorHandle v = tensor_create(data, shape, 1, 0);
	TensorHandle r = tensor_subtract_scalar_inplace(v, 5.0);
	cr_assert_eq((void*)r, (void*)v, "in-place op returns the same handle");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 5.0, 1e-12);
	cr_assert_float_eq(out[1], 15.0, 1e-12);
	cr_assert_float_eq(out[2], 25.0, 1e-12);
}

Test(core_lifecycle, mul_elementwise_alias) {
	/* Back-compat alias for tensor_mul (Hadamard product). */
	double a[] = {1.0, 2.0, 3.0};
	double b[] = {4.0, 5.0, 6.0};
	int shape[] = {3};
	TensorHandle ha = tensor_create(a, shape, 1, 0);
	TensorHandle hb = tensor_create(b, shape, 1, 0);
	TensorHandle prod = tensor_mul_elementwise(ha, hb);
	double out[3];
	tensor_to_doubles(prod, out);
	cr_assert_float_eq(out[0], 4.0, 1e-12);
	cr_assert_float_eq(out[1], 10.0, 1e-12);
	cr_assert_float_eq(out[2], 18.0, 1e-12);
}

Test(core_lifecycle, sum_all_alias) {
	/* Back-compat alias for tensor_sum (full reduction to a scalar). */
	double a[] = {1.0, 2.0, 3.0, 4.0};
	int shape[] = {4};
	TensorHandle ha = tensor_create(a, shape, 1, 0);
	TensorHandle s = tensor_sum_all(ha);
	cr_assert_eq(tensor_dim(s), 0);
	cr_assert_float_eq(tensor_item(s), 10.0, 1e-12, "sum_all should total all elements");
}

Test(core_lifecycle, batch_empty_returns_rank1_zero) {
	/* count == 0: early-out returns a [0] tensor, no input handles read. */
	TensorHandle batched = tensor_batch(NULL, 0);
	cr_assert_eq(tensor_dim(batched), 1);
	cr_assert_eq(tensor_size(batched, 0), 0);
	cr_assert_eq(tensor_numel(batched), 0);
}

Test(core_lifecycle, batch_stacks_via_ptr_array) {
	/* The single-FFI collation path Idris DataStream.collate wires:
	   stage B handles into a TensorHandle* via the ptr-array helpers,
	   then tensor_batch stacks them along a new leading axis → [B, ...].
	   Row-major: batching [1,2,3] and [4,5,6] yields [[1,2,3],[4,5,6]]. */
	double d0[] = {1.0, 2.0, 3.0};
	double d1[] = {4.0, 5.0, 6.0};
	int shape[] = {3};
	TensorHandle a = tensor_create(d0, shape, 1, 0);
	TensorHandle b = tensor_create(d1, shape, 1, 0);
	void** arr = tensor_ptr_array_alloc(2);
	tensor_ptr_array_set_return(arr, 0, a);
	tensor_ptr_array_set_return(arr, 1, b);
	TensorHandle batched = tensor_batch(arr, 2);
	tensor_ptr_array_free(arr);
	cr_assert_eq(tensor_dim(batched), 2);
	cr_assert_eq(tensor_size(batched, 0), 2);
	cr_assert_eq(tensor_size(batched, 1), 3);
	double out[6];
	tensor_to_doubles(batched, out);
	double expect[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expect[i], 1e-12, "tensor_batch row-major [2,3] cell %d", i);
}

/* ----------------------------------------------------------------------
   F32 coverage — clone / subtract_scalar_inplace / batch F32 branches.
   F32 readback (tensor_to_doubles) carries ~1e-6 error; assert at 1e-5.
   ---------------------------------------------------------------------- */

Test(core_lifecycle, clone_f32_vector) {
	/* clone.c F32 non-scalar path (lines 22-24): an F32 input clones via the
	   float storage memcpy + make_tensor_arena_f32, preserving values and the
	   F32 dtype tag. */
	double data[] = {1.5, 2.5, 3.5};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(data, 3), 0, 0, 14);
	TensorHandle b = tensor_clone(a);
	cr_assert_str_eq(tensor_dtype_name(b), "F32", "F32 clone keeps the F32 dtype tag");
	cr_assert_eq(tensor_numel(b), 3);
	cr_assert_eq(tensor_dim(b), 1);
	cr_assert_neq((void*)a, (void*)b, "clone must be a new handle");
	double out[3];
	tensor_to_doubles(b, out);
	cr_assert_float_eq(out[0], 1.5, 1e-5, "F32 clone cell 0");
	cr_assert_float_eq(out[1], 2.5, 1e-5, "F32 clone cell 1");
	cr_assert_float_eq(out[2], 3.5, 1e-5, "F32 clone cell 2");
}

Test(core_lifecycle, clone_f32_scalar_keeps_dtype) {
	/* clone.c scalar path: rank-0 F32 clone routes through make_scalar_f32. */
	TensorHandle a = tensor_create_scalar_streamed(9.25, 0, 0, 14);
	TensorHandle b = tensor_clone(a);
	cr_assert_str_eq(tensor_dtype_name(b), "F32");
	cr_assert_eq(tensor_dim(b), 0);
	cr_assert_float_eq(tensor_item(b), 9.25, 1e-5, "F32 scalar clone value");
}

Test(core_lifecycle, subtract_scalar_inplace_mutates_f32) {
	/* lifecycle_ext.c F32 branch (lines 71-74): per-element float subtraction
	   in place, returns the same handle. */
	double data[] = {10.0, 20.0, 30.0};
	TensorHandle v = tensor_create_1d_streamed(3, hcopy(data, 3), 0, 0, 14);
	TensorHandle r = tensor_subtract_scalar_inplace(v, 5.0);
	cr_assert_eq((void*)r, (void*)v, "in-place op returns the same handle");
	cr_assert_str_eq(tensor_dtype_name(r), "F32");
	double out[3];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 5.0, 1e-5);
	cr_assert_float_eq(out[1], 15.0, 1e-5);
	cr_assert_float_eq(out[2], 25.0, 1e-5);
}

Test(core_lifecycle, batch_stacks_f32) {
	/* lifecycle_ext.c F32 branch (lines 110-116): stacking F32 inputs goes
	   through the arena float allocator and preserves the F32 dtype tag. */
	double d0[] = {1.0, 2.0, 3.0};
	double d1[] = {4.0, 5.0, 6.0};
	TensorHandle a = tensor_create_1d_streamed(3, hcopy(d0, 3), 0, 0, 14);
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(d1, 3), 0, 0, 14);
	void** arr = tensor_ptr_array_alloc(2);
	tensor_ptr_array_set_return(arr, 0, a);
	tensor_ptr_array_set_return(arr, 1, b);
	TensorHandle batched = tensor_batch(arr, 2);
	tensor_ptr_array_free(arr);
	cr_assert_str_eq(tensor_dtype_name(batched), "F32", "F32 batch keeps the F32 dtype tag");
	cr_assert_eq(tensor_dim(batched), 2);
	cr_assert_eq(tensor_size(batched, 0), 2);
	cr_assert_eq(tensor_size(batched, 1), 3);
	double out[6];
	tensor_to_doubles(batched, out);
	double expect[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	for (int i = 0; i < 6; i++)
		cr_assert_float_eq(out[i], expect[i], 1e-5, "F32 tensor_batch cell %d", i);
}

Test(core_lifecycle, batch_mixed_dtype_aborts, .signal = SIGABRT) {
	/* lifecycle_ext.c line 104: a second input whose dtype_tag differs from
	   the first triggers tape_abort_mixed_dtype -> abort(). Stack an F64 and
	   an F32 of identical shape so only the dtype mismatch fires the abort. */
	double d0[] = {1.0, 2.0, 3.0};
	double d1[] = {4.0, 5.0, 6.0};
	int shape[] = {3};
	TensorHandle a = tensor_create(d0, shape, 1, 0);                       /* F64 first */
	TensorHandle b = tensor_create_1d_streamed(3, hcopy(d1, 3), 0, 0, 14); /* F32 mismatch */
	void** arr = tensor_ptr_array_alloc(2);
	tensor_ptr_array_set_return(arr, 0, a);
	tensor_ptr_array_set_return(arr, 1, b);
	tensor_batch(arr, 2); /* expected: SIGABRT */
}

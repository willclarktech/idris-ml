/* training/per_dtype_aliases.c — per-dtype creator aliases for the
 * backend.h ABI surface.
 *
 * backend.h declares per-dtype `_f32` / `_f64` creator + cast symbols;
 * every backend in the multi-link dylib exports the full prototype set.
 * On tape:
 *   - The `_f64` variants delegate to the unsuffixed F64 creators
 *     (identity — F64 is tape's lingua-franca path). These are the
 *     bare-ABI construction API the C unit-test suite calls directly
 *     (tensor_create_2d_f64 / tensor_create_param_*_f64 / ...).
 *   - The `_f32` variants are abort stubs: tape has no fp32 *bare*-ABI
 *     arena. Real tape F32 storage exists, but only via the unified
 *     `tensor_create_<shape>_streamed(..., int dtag)` entry points
 *     (tape_arena_f32_from_doubles / tape_persistent_f32_from_doubles),
 *     which the typed Idris surface routes through exclusively. The bare
 *     `_f32` symbols have no caller on tape; the abort diagnostic is
 *     reachable only via direct C linkage, and the symbols exist solely
 *     for backend.h ABI completeness across the multi-link dylib.
 *
 * F64 cast is observational identity (preserves autograd-through-cast);
 * F32 cast routes through `tensor_cast_dtype_streamed(src, _, 0)` only.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../backend.h"

// GCOVR_EXCL_START — abort path; covered by death tests in test_dtype_aliases.c
// (tape_f32_aliases.*_aborts). abort() skips the gcov flush in the forked
// Criterion child, so the lines can't register as covered despite firing.
static TensorHandle tape_f32_unsupported(const char* sym) {
	fprintf(stderr,
	        "[tape backend] %s called but tape has no fp32 arena. "
	        "Bind your code to F64 on tape, or build with BACKEND=mlx / torch.\n",
	        sym);
	// NOLINTNEXTLINE(misc-include-cleaner): macOS SDK: abort via _abort.h umbrella
	abort();
}
// GCOVR_EXCL_STOP

TensorHandle tensor_create_scalar_f64(double v, int rg) {
	return tensor_create_scalar(v, rg);
}
TensorHandle tensor_create_f64(double* d, int* s, int r, int rg) {
	return tensor_create(d, s, r, rg);
}
TensorHandle tensor_create_1d_f64(int n, double* d, int rg) {
	int shape[] = {n};
	TensorHandle t = tensor_create(d, shape, 1, rg);
	free(d); /* tensor_create copies data into arena; free the original */
	return t;
}
TensorHandle tensor_create_2d_f64(int rows, int cols, double* d, int rg) {
	return tensor_create_2d(rows, cols, d, rg);
}
/* tensor_create_{param,state}_{Nd}_f64 live in training/param_create.c
   (the F64-default path is the only path tape supports for these). */

// GCOVR_EXCL_START — every _f32 bare stub funnels into tape_f32_unsupported's
// abort(); proven to fire by the SIGABRT death tests in test_dtype_aliases.c
// (tape_f32_aliases suite). The fork-and-abort skips the gcov flush, so the
// stub bodies can't register as covered.
TensorHandle tensor_create_scalar_f32(double v, int rg) {
	(void)v;
	(void)rg;
	return tape_f32_unsupported("tensor_create_scalar_f32");
}
TensorHandle tensor_create_f32(double* d, int* s, int r, int rg) {
	(void)d;
	(void)s;
	(void)r;
	(void)rg;
	return tape_f32_unsupported("tensor_create_f32");
}
TensorHandle tensor_create_1d_f32(int n, double* d, int rg) {
	(void)n;
	(void)d;
	(void)rg;
	return tape_f32_unsupported("tensor_create_1d_f32");
}
TensorHandle tensor_create_2d_f32(int rows, int cols, double* d, int rg) {
	(void)rows;
	(void)cols;
	(void)d;
	(void)rg;
	return tape_f32_unsupported("tensor_create_2d_f32");
}
TensorHandle tensor_create_param_1d_f32(int n, double* d) {
	(void)n;
	(void)d;
	return tape_f32_unsupported("tensor_create_param_1d_f32");
}
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* d) {
	(void)rows;
	(void)cols;
	(void)d;
	return tape_f32_unsupported("tensor_create_param_2d_f32");
}
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* d) {
	(void)d0;
	(void)d1;
	(void)d2;
	(void)d;
	return tape_f32_unsupported("tensor_create_param_3d_f32");
}
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* d) {
	(void)d0;
	(void)d1;
	(void)d2;
	(void)d3;
	(void)d;
	return tape_f32_unsupported("tensor_create_param_4d_f32");
}
TensorHandle tensor_create_state_1d_f32(int n, double* d) {
	(void)n;
	(void)d;
	return tape_f32_unsupported("tensor_create_state_1d_f32");
}
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* d) {
	(void)rows;
	(void)cols;
	(void)d;
	return tape_f32_unsupported("tensor_create_state_2d_f32");
}
// GCOVR_EXCL_STOP

/* F64 cast is observational identity (no new tape op — gradients flow
   through the source's tape entry). F32 cast aborts here; the real
   path is `tensor_cast_dtype_streamed(src, _, 0)`. */
TensorHandle tensor_cast_dtype_f64(TensorHandle src) {
	return src;
}
// GCOVR_EXCL_START — abort stub; covered by tape_f32_aliases.cast_dtype_f32_aborts
TensorHandle tensor_cast_dtype_f32(TensorHandle src) {
	(void)src;
	return tape_f32_unsupported("tensor_cast_dtype_f32");
}
// GCOVR_EXCL_STOP

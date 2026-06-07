/* training/per_dtype_legacy.c — per-dtype creator aliases (pre-streamed).
 *
 * These per-dtype `_f32` / `_f64` symbols predate the
 * unified `tensor_create_<shape>_streamed(..., int dtag)` entry points
 * that landed with the FFI tag-dispatch unification (2026-05-22). The
 * typed Idris surface routes through the unified streamed symbols
 * exclusively; these per-dtype variants are kept only for `backend.h`
 * ABI completeness across the multi-link dylib (every backend must
 * export every prototype declared in backend.h).
 *
 * The _f64 variants delegate to the unsuffixed F64 creators — identity,
 * F64 is the lingua-franca path. The _f32 variants are left as abort
 * stubs even though tape now has *real* F32 storage via
 * `tape_arena_f32_from_doubles` / `tape_persistent_f32_from_doubles`
 * (used by the unified streamed path); no caller reaches the legacy
 * `_f32` symbols any more — the abort diagnostic is reachable only via
 * direct C linkage.
 *
 * F64 cast is observational identity (preserves autograd-through-cast);
 * F32 cast routes through `tensor_cast_dtype_streamed(src, _, 0)` only.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../backend.h"

static TensorHandle tape_f32_unsupported(const char* sym) {
	fprintf(stderr,
	        "[tape backend] %s called but tape has no fp32 arena. "
	        "Bind your code to F64 on tape, or build with BACKEND=mlx / torch.\n",
	        sym);
	// NOLINTNEXTLINE(misc-include-cleaner): macOS SDK: abort via _abort.h umbrella
	abort();
}

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

/* F64 cast is observational identity (no new tape op — gradients flow
   through the source's tape entry). F32 cast aborts here; the real
   path is `tensor_cast_dtype_streamed(src, _, 0)`. */
TensorHandle tensor_cast_dtype_f64(TensorHandle src) {
	return src;
}
TensorHandle tensor_cast_dtype_f32(TensorHandle src) {
	(void)src;
	return tape_f32_unsupported("tensor_cast_dtype_f32");
}

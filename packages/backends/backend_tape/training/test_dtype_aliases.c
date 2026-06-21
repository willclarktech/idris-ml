/* Criterion suite for tape's dtag-streamed creators + per-dtype aliases.
 *
 * Covers three product TUs on the tape lane:
 *   - training/dtype_dispatch.c     — tape_create_*_dtag (F64=15 / F32=14 /
 *                                     lingua-franca retag, e.g. I32=10) + cast.
 *   - shared/training/dtype_streamed.c — the tensor_create_*_streamed shells
 *                                     that forward to g_active_port, plus the
 *                                     fused-init _normal_ / _const_ wrappers
 *                                     and tensor_set_init_seed_streamed.
 *   - training/per_dtype_aliases.c  — the bare-ABI _f32 abort stubs (death
 *                                     tests; bodies GCOVR_EXCL'd in source).
 *
 * The streamed/dtag creators FREE their `data` argument (mirroring the *_f64
 * creator contract), so every input buffer is hcopy'd. F32 readback
 * carries ~1e-6 error, so F32 asserts use an explicit 1e-5 tolerance.
 *
 * dtag codes (see test_dtype_scaffolding.c / tape_tag_from_dtag): F64=15,
 * F32=14, I32=10, BF16=17.
 */

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

/* Tape-specific: the _f32 bare aliases abort ONLY on tape (torch/mlx have a
   real fp32 path), and the lingua-franca dtag retag is a tape behaviour. The
   colocated backend test build links all backends' test_*.c into one binary
   keyed by the PRIMARY backend, so guard the whole suite to the tape build. */
#ifdef BACKEND_TAPE

/* Heap-copy a stack array — the dtag creators take ownership and free it. */

/* ----------------------------------------------------------------------
   dtype_dispatch.c — F64 / F32 / lingua-franca branches of every creator.
   Streaming through tensor_create_*_streamed exercises BOTH the shared
   wrapper (dtype_streamed.c) and the tape dispatcher (dtype_dispatch.c).
   ---------------------------------------------------------------------- */

Test(tape_dtype_dispatch, create_scalar_all_dtags) {
	param_clear();
	/* F64 identity (dtag 15). */
	TensorHandle s64 = tensor_create_scalar_streamed(2.5, 0, 0, 15);
	cr_assert_str_eq(tensor_dtype_name(s64), "F64");
	cr_assert_float_eq(tensor_item(s64), 2.5, 1e-12);
	/* F32 real-storage scalar (dtag 14). */
	TensorHandle s32 = tensor_create_scalar_streamed(2.5, 0, 0, 14);
	cr_assert_str_eq(tensor_dtype_name(s32), "F32");
	cr_assert_float_eq(tensor_item(s32), 2.5, 1e-5);
	/* I32 lingua-franca retag (dtag 10). */
	TensorHandle si = tensor_create_scalar_streamed(7.0, 0, 0, 10);
	cr_assert_str_eq(tensor_dtype_name(si), "I32");
	cr_assert_float_eq(tensor_item(si), 7.0, 1e-10);
	param_clear();
}

Test(tape_dtype_dispatch, create_nd_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	int shape[] = {2, 3};
	/* F64 (dtag 15). */
	TensorHandle t64 = tensor_create_streamed(hcopy(v, 6), shape, 2, 0, 0, 15);
	cr_assert_str_eq(tensor_dtype_name(t64), "F64");
	cr_assert_eq(tensor_numel(t64), 6);
	/* F32 arena (dtag 14) — exercises the inline f32 build in tape_create_dtag. */
	TensorHandle t32 = tensor_create_streamed(hcopy(v, 6), shape, 2, 0, 0, 14);
	cr_assert_str_eq(tensor_dtype_name(t32), "F32");
	double o32[6];
	tensor_to_doubles(t32, o32);
	cr_assert_float_eq(o32[5], 6.0, 1e-5);
	/* I32 lingua-franca retag (dtag 10) — covers tape_create_dtag retag arm. */
	TensorHandle ti = tensor_create_streamed(hcopy(v, 6), shape, 2, 0, 0, 10);
	cr_assert_str_eq(tensor_dtype_name(ti), "I32");
	double oi[6];
	tensor_to_doubles(ti, oi);
	cr_assert_float_eq(oi[2], 3.0, 1e-10);
	param_clear();
}

Test(tape_dtype_dispatch, create_1d_lingua_franca) {
	param_clear();
	double v[] = {1.5, 2.5, 3.5};
	/* I32 retag arm of tape_create_1d_dtag. */
	TensorHandle ti = tensor_create_1d_streamed(3, hcopy(v, 3), 0, 0, 10);
	cr_assert_str_eq(tensor_dtype_name(ti), "I32");
	param_clear();
}

Test(tape_dtype_dispatch, create_2d_lingua_franca) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0, 4.0};
	/* I32 retag arm of tape_create_2d_dtag (dispatch.c:118). */
	TensorHandle ti = tensor_create_2d_streamed(2, 2, hcopy(v, 4), 0, 0, 10);
	cr_assert_str_eq(tensor_dtype_name(ti), "I32");
	cr_assert_eq(tensor_numel(ti), 4);
	param_clear();
}

Test(tape_dtype_dispatch, create_param_1d_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0};
	TensorHandle p64 = tensor_create_param_1d_streamed(3, hcopy(v, 3), 0, 15);
	cr_assert_str_eq(tensor_dtype_name(p64), "F64");
	TensorHandle p32 = tensor_create_param_1d_streamed(3, hcopy(v, 3), 0, 14);
	cr_assert_str_eq(tensor_dtype_name(p32), "F32");
	/* I32 retag arm of tape_create_param_1d_dtag (dispatch.c:127). */
	TensorHandle pi = tensor_create_param_1d_streamed(3, hcopy(v, 3), 0, 10);
	cr_assert_str_eq(tensor_dtype_name(pi), "I32");
	param_clear();
}

Test(tape_dtype_dispatch, create_param_2d_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle p32 = tensor_create_param_2d_streamed(2, 2, hcopy(v, 4), 0, 14);
	cr_assert_str_eq(tensor_dtype_name(p32), "F32");
	/* I32 retag arm of tape_create_param_2d_dtag (dispatch.c:136). */
	TensorHandle pi = tensor_create_param_2d_streamed(2, 2, hcopy(v, 4), 0, 10);
	cr_assert_str_eq(tensor_dtype_name(pi), "I32");
	param_clear();
}

Test(tape_dtype_dispatch, create_param_3d_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle p64 = tensor_create_param_3d_streamed(2, 2, 2, hcopy(v, 8), 0, 15);
	cr_assert_str_eq(tensor_dtype_name(p64), "F64");
	TensorHandle p32 = tensor_create_param_3d_streamed(2, 2, 2, hcopy(v, 8), 0, 14);
	cr_assert_str_eq(tensor_dtype_name(p32), "F32");
	/* I32 retag arm of tape_create_param_3d_dtag (dispatch.c:145). */
	TensorHandle pi = tensor_create_param_3d_streamed(2, 2, 2, hcopy(v, 8), 0, 10);
	cr_assert_str_eq(tensor_dtype_name(pi), "I32");
	param_clear();
}

Test(tape_dtype_dispatch, create_param_4d_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	/* F64 (dispatch.c:149), F32 (150-152), I32 retag (154). */
	TensorHandle p64 = tensor_create_param_4d_streamed(2, 2, 1, 2, hcopy(v, 8), 0, 15);
	cr_assert_str_eq(tensor_dtype_name(p64), "F64");
	TensorHandle p32 = tensor_create_param_4d_streamed(2, 2, 1, 2, hcopy(v, 8), 0, 14);
	cr_assert_str_eq(tensor_dtype_name(p32), "F32");
	cr_assert_eq(tensor_numel(p32), 8);
	TensorHandle pi = tensor_create_param_4d_streamed(2, 2, 1, 2, hcopy(v, 8), 0, 10);
	cr_assert_str_eq(tensor_dtype_name(pi), "I32");
	param_clear();
}

Test(tape_dtype_dispatch, create_state_1d_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0};
	TensorHandle s64 = tensor_create_state_1d_streamed(3, hcopy(v, 3), 0, 15);
	cr_assert_str_eq(tensor_dtype_name(s64), "F64");
	TensorHandle s32 = tensor_create_state_1d_streamed(3, hcopy(v, 3), 0, 14);
	cr_assert_str_eq(tensor_dtype_name(s32), "F32");
	/* I32 retag arm of tape_create_state_1d_dtag (dispatch.c:163). */
	TensorHandle si = tensor_create_state_1d_streamed(3, hcopy(v, 3), 0, 10);
	cr_assert_str_eq(tensor_dtype_name(si), "I32");
	param_clear();
}

Test(tape_dtype_dispatch, create_state_2d_all_dtags) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0, 4.0};
	/* F64 (dispatch.c:167), F32 (168-170), I32 retag (172). */
	TensorHandle s64 = tensor_create_state_2d_streamed(2, 2, hcopy(v, 4), 0, 15);
	cr_assert_str_eq(tensor_dtype_name(s64), "F64");
	TensorHandle s32 = tensor_create_state_2d_streamed(2, 2, hcopy(v, 4), 0, 14);
	cr_assert_str_eq(tensor_dtype_name(s32), "F32");
	TensorHandle si = tensor_create_state_2d_streamed(2, 2, hcopy(v, 4), 0, 10);
	cr_assert_str_eq(tensor_dtype_name(si), "I32");
	param_clear();
}

/* Cast a rank-0 F64 scalar to a lingua-franca dtype (I32): exercises the
   scalar non-F32 arm of tape_cast_dtype_dtag (dispatch.c:226-228). */
Test(tape_dtype_dispatch, cast_scalar_lingua_franca) {
	param_clear();
	TensorHandle s = tensor_create_scalar_streamed(5.0, 0, 0, 15);
	TensorHandle ci = tensor_cast_dtype_streamed(s, 0, 10); /* F64 -> I32 (scalar) */
	cr_assert_str_eq(tensor_dtype_name(ci), "I32");
	cr_assert_float_eq(tensor_item(ci), 5.0, 1e-10);
	param_clear();
}

/* ----------------------------------------------------------------------
   dtype_streamed.c — fused param create+init wrappers (all wired on tape)
   and the seed setter.
   ---------------------------------------------------------------------- */

Test(tape_dtype_streamed, fused_init_normal_all_ranks) {
	param_clear();
	tensor_set_init_seed_streamed(123ULL, 0); /* dtype_streamed.c:163-165 */
	TensorHandle n1 = tensor_create_param_1d_normal_streamed(4, 0.0, 0.02, 0, 15);
	cr_assert_eq(tensor_numel(n1), 4);
	cr_assert_str_eq(tensor_dtype_name(n1), "F64");
	TensorHandle n2 = tensor_create_param_2d_normal_streamed(2, 3, 0.0, 0.02, 0, 15);
	cr_assert_eq(tensor_numel(n2), 6);
	TensorHandle n3 = tensor_create_param_3d_normal_streamed(2, 2, 2, 0.0, 0.02, 0, 15);
	cr_assert_eq(tensor_numel(n3), 8);
	TensorHandle n4 = tensor_create_param_4d_normal_streamed(2, 1, 2, 2, 0.0, 0.02, 0, 15);
	cr_assert_eq(tensor_numel(n4), 8);
	param_clear();
}

Test(tape_dtype_streamed, fused_init_const_all_ranks) {
	param_clear();
	TensorHandle c1 = tensor_create_param_1d_const_streamed(3, 0.5, 0, 15);
	cr_assert_eq(tensor_numel(c1), 3);
	cr_assert_float_eq(tensor_item_1d(c1, 0), 0.5, 1e-12);
	TensorHandle c2 = tensor_create_param_2d_const_streamed(2, 2, 0.5, 0, 15);
	cr_assert_eq(tensor_numel(c2), 4);
	TensorHandle c3 = tensor_create_param_3d_const_streamed(2, 2, 2, 0.5, 0, 15);
	cr_assert_eq(tensor_numel(c3), 8);
	TensorHandle c4 = tensor_create_param_4d_const_streamed(2, 1, 2, 2, 0.5, 0, 15);
	cr_assert_eq(tensor_numel(c4), 8);
	param_clear();
}

/* ----------------------------------------------------------------------
   per_dtype_aliases.c — the bare-ABI _f32 stubs all abort on tape (no fp32
   *bare* arena; real F32 routes through the _streamed dtag dispatchers
   above). One death test per stub; the abort-only bodies are GCOVR_EXCL'd
   in source (abort() skips the gcov flush in the forked child).
   ---------------------------------------------------------------------- */

Test(tape_f32_aliases, create_scalar_f32_aborts, .signal = SIGABRT) {
	tensor_create_scalar_f32(1.0, 0);
}

Test(tape_f32_aliases, create_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0};
	int s[] = {1};
	tensor_create_f32(hcopy(v, 1), s, 1, 0);
}

Test(tape_f32_aliases, create_1d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0};
	tensor_create_1d_f32(2, hcopy(v, 2), 0);
}

Test(tape_f32_aliases, create_2d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0, 3.0, 4.0};
	tensor_create_2d_f32(2, 2, hcopy(v, 4), 0);
}

Test(tape_f32_aliases, create_param_1d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0};
	tensor_create_param_1d_f32(2, hcopy(v, 2));
}

Test(tape_f32_aliases, create_param_2d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0, 3.0, 4.0};
	tensor_create_param_2d_f32(2, 2, hcopy(v, 4));
}

Test(tape_f32_aliases, create_param_3d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	tensor_create_param_3d_f32(2, 2, 2, hcopy(v, 8));
}

Test(tape_f32_aliases, create_param_4d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	tensor_create_param_4d_f32(2, 2, 1, 2, hcopy(v, 8));
}

Test(tape_f32_aliases, create_state_1d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0};
	tensor_create_state_1d_f32(2, hcopy(v, 2));
}

Test(tape_f32_aliases, create_state_2d_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0, 3.0, 4.0};
	tensor_create_state_2d_f32(2, 2, hcopy(v, 4));
}

Test(tape_f32_aliases, cast_dtype_f32_aborts, .signal = SIGABRT) {
	double v[] = {1.0, 2.0};
	int s[] = {2};
	TensorHandle src = tensor_create(v, s, 1, 0);
	tensor_cast_dtype_f32(src);
}

#endif /* BACKEND_TAPE */

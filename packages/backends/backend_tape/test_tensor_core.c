/* Criterion suite for backend_tape/tensor.c — the element-size +
 * ABI<->internal dtype-tag translation + lingua-franca rounding helpers.
 *
 * These three helpers (tape_elem_size / tape_tag_from_dtag /
 * tape_round_to_dtype) are not part of the public FFI (backend.h); they
 * are internal tape utilities exercised indirectly by the dtag-streamed
 * creators. We re-declare them as externs (the test_grad_typed.c pattern)
 * and drive every dtype arm directly, since the streamed-create path only
 * happens to touch a subset on the F64+F32 tape lane.
 *
 * The internal DT_* tags are dense (F64=0, F32=1, ...) per tensor.h; the
 * ABI dtag codes are kind-major (Bool=1, U8=4, I8..I64=8..11, F16=13,
 * F32=14, F64=15, BF16=17, Binary=24, Ternary=25).
 */

#include <criterion/criterion.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include "backend.h"

#ifdef BACKEND_TAPE

/* Mirror the internal DT_* enum (tensor.h) so we can name the tags the
   helpers switch on. Dense layout, F64 = 0. */
enum {
	TC_DT_F64 = 0,
	TC_DT_F32,
	TC_DT_BF16,
	TC_DT_F16,
	TC_DT_I8,
	TC_DT_I16,
	TC_DT_I32,
	TC_DT_I64,
	TC_DT_U8,
	TC_DT_BOOL,
	TC_DT_BINARY,
	TC_DT_TERNARY
};

extern size_t tape_elem_size(int tag);
extern int tape_tag_from_dtag(int dtag);
extern double tape_round_to_dtype(double v, int tag);

/* --------------------------------------------------------------------
 * tape_elem_size — one assert per non-default storage width plus the
 * sub-byte (0) and unknown-tag (F64 fallback) arms.
 * ------------------------------------------------------------------ */
Test(tape_tensor_core, elem_size_all_arms) {
	cr_assert_eq(tape_elem_size(TC_DT_F64), sizeof(double), "F64 -> 8");
	cr_assert_eq(tape_elem_size(TC_DT_F32), sizeof(float), "F32 -> 4");
	cr_assert_eq(tape_elem_size(TC_DT_BF16), sizeof(uint16_t), "BF16 -> 2");
	cr_assert_eq(tape_elem_size(TC_DT_F16), sizeof(uint16_t), "F16 -> 2");
	cr_assert_eq(tape_elem_size(TC_DT_I8), sizeof(int8_t), "I8 -> 1");
	cr_assert_eq(tape_elem_size(TC_DT_U8), sizeof(int8_t), "U8 -> 1");
	cr_assert_eq(tape_elem_size(TC_DT_BOOL), sizeof(int8_t), "BOOL -> 1");
	cr_assert_eq(tape_elem_size(TC_DT_I16), sizeof(int16_t), "I16 -> 2");
	cr_assert_eq(tape_elem_size(TC_DT_I32), sizeof(int32_t), "I32 -> 4");
	cr_assert_eq(tape_elem_size(TC_DT_I64), sizeof(int64_t), "I64 -> 8");
	/* Sub-byte tags report 0 (caller consults tape_packed_bytes). */
	cr_assert_eq(tape_elem_size(TC_DT_BINARY), 0u, "BINARY -> 0 (sub-byte)");
	cr_assert_eq(tape_elem_size(TC_DT_TERNARY), 0u, "TERNARY -> 0 (sub-byte)");
	/* Unknown tag falls through to the F64-sized default. */
	cr_assert_eq(tape_elem_size(999), sizeof(double), "unknown -> 8 (F64 default)");
}

/* --------------------------------------------------------------------
 * tape_tag_from_dtag — every valid ABI dtag maps to its dense DT_* tag.
 * ------------------------------------------------------------------ */
Test(tape_tensor_core, tag_from_dtag_all_valid) {
	cr_assert_eq(tape_tag_from_dtag(1), TC_DT_BOOL, "dtag 1 -> BOOL");
	cr_assert_eq(tape_tag_from_dtag(4), TC_DT_U8, "dtag 4 -> U8");
	cr_assert_eq(tape_tag_from_dtag(8), TC_DT_I8, "dtag 8 -> I8");
	cr_assert_eq(tape_tag_from_dtag(9), TC_DT_I16, "dtag 9 -> I16");
	cr_assert_eq(tape_tag_from_dtag(10), TC_DT_I32, "dtag 10 -> I32");
	cr_assert_eq(tape_tag_from_dtag(11), TC_DT_I64, "dtag 11 -> I64");
	cr_assert_eq(tape_tag_from_dtag(13), TC_DT_F16, "dtag 13 -> F16");
	cr_assert_eq(tape_tag_from_dtag(14), TC_DT_F32, "dtag 14 -> F32");
	cr_assert_eq(tape_tag_from_dtag(15), TC_DT_F64, "dtag 15 -> F64");
	cr_assert_eq(tape_tag_from_dtag(17), TC_DT_BF16, "dtag 17 -> BF16");
	cr_assert_eq(tape_tag_from_dtag(24), TC_DT_BINARY, "dtag 24 -> BINARY");
	cr_assert_eq(tape_tag_from_dtag(25), TC_DT_TERNARY, "dtag 25 -> TERNARY");
}

/* Invalid dtag hits the loud-abort default arm. Death test: fork-isolated
   by Criterion; the abort body is otherwise unreachable from valid input. */
Test(tape_tensor_core, tag_from_dtag_invalid_aborts, .signal = SIGABRT) {
	(void)tape_tag_from_dtag(99);
}

/* --------------------------------------------------------------------
 * tape_round_to_dtype — drive every precision-rounding arm.
 * ------------------------------------------------------------------ */
Test(tape_tensor_core, round_to_dtype_arms) {
	/* F32: 0.1 is not representable in binary32 — round trip drops bits. */
	double rf32 = tape_round_to_dtype(0.1, TC_DT_F32);
	cr_assert_float_eq(rf32, (double)(float)0.1, 1e-12, "F32 round matches (float)0.1");
	/* BF16 / F16 round to lower-precision floats; exact integers survive. */
	cr_assert_float_eq(tape_round_to_dtype(2.0, TC_DT_BF16), 2.0, 1e-6, "BF16 keeps 2.0");
	cr_assert_float_eq(tape_round_to_dtype(2.0, TC_DT_F16), 2.0, 1e-6, "F16 keeps 2.0");
	/* Integer dtypes truncate toward zero. */
	cr_assert_float_eq(tape_round_to_dtype(3.9, TC_DT_I8), 3.0, 1e-12, "I8 truncates 3.9 -> 3");
	cr_assert_float_eq(tape_round_to_dtype(-3.9, TC_DT_I16), -3.0, 1e-12,
	                   "I16 truncates -3.9 -> -3");
	cr_assert_float_eq(tape_round_to_dtype(40000.7, TC_DT_I32), 40000.0, 1e-12, "I32 truncates");
	cr_assert_float_eq(tape_round_to_dtype(7.5, TC_DT_I64), 7.0, 1e-12, "I64 truncates 7.5 -> 7");
	/* U8 wraps the low byte of the truncated value. */
	cr_assert_float_eq(tape_round_to_dtype(257.0, TC_DT_U8), 1.0, 1e-12, "U8 wraps 257 -> 1");
	/* BOOL collapses to 0/1. */
	cr_assert_float_eq(tape_round_to_dtype(0.0, TC_DT_BOOL), 0.0, 1e-12, "BOOL 0.0 -> 0");
	cr_assert_float_eq(tape_round_to_dtype(5.0, TC_DT_BOOL), 1.0, 1e-12, "BOOL 5.0 -> 1");
	/* Default (F64 / unknown) is identity. */
	cr_assert_float_eq(tape_round_to_dtype(3.14159, TC_DT_F64), 3.14159, 1e-12, "F64 identity");
	cr_assert_float_eq(tape_round_to_dtype(3.14159, 999), 3.14159, 1e-12, "unknown -> identity");
}

#endif /* BACKEND_TAPE */

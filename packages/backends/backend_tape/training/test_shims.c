/* Criterion suite for tape's system/debug shims + the shared *_return FFI
 * wrappers.
 *
 * Covers two product TUs on the tape lane:
 *   - backend_tape/training/shims.c        — backend_reset_for_eval (with the
 *     param re-registration loop), backend_release_all_persistent (the
 *     three-phase teardown), the tape_size / g_tape_peak live-count probes,
 *     backend_name, the mlx_compile no-op accessors, and tensor_print's
 *     rank-0 + rank-N dump.
 *   - shared/training/ffi_shims.c          — every `*_return` shim: backward,
 *     param_register, param_register_buffer, zero_all_grads, to_doubles,
 *     backward_return_loss, idrisml_seq, reset_for_eval, profile reset/report.
 *
 * Each Criterion Test() runs in its own forked child, so the destructive
 * backend_release_all_persistent teardown can't corrupt sibling tests.
 *
 * tensor_create copies (does not free) its data argument, so stack arrays are
 * passed directly. F32 isn't exercised here: tape's bare _f32 creators abort
 * (covered in test_dtype_aliases.c) and none of the shim lines branch on dtype.
 */

#include <criterion/criterion.h>
#include "backend.h"

/* ----------------------------------------------------------------------
   shims.c — System: backend_reset_for_eval + backend_release_all_persistent.
   ---------------------------------------------------------------------- */

/* Re-registration loop (shims.c:25-30): register a param, reset for eval, and
   confirm the param survives the tape reset and its grad was zeroed. */
Test(tape_shims, reset_for_eval_reregisters_params) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0};
	int shape[] = {3};
	TensorHandle p = tensor_create(v, shape, 1, 1);
	param_register("w", p);
	cr_assert_eq(param_count(), 1);

	/* Drive a backward so the param has a non-zero grad to be zeroed. */
	TensorHandle loss = tensor_sum(p);
	tensor_backward(loss);

	backend_reset_for_eval();

	/* Param still registered after the tape reset. */
	cr_assert_eq(param_count(), 1);
	/* Grad zeroed by the memset in the re-registration loop. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12);
	param_clear();
}

#ifdef BACKEND_TAPE
/* Three-phase teardown (shims.c:60-72): frees every registered param, clears
   the registry, and frees the arena. Runs in a forked child, so the arena
   free can't strand sibling tests. */
Test(tape_shims, release_all_persistent_clears_registry) {
	param_clear();
	double v[] = {4.0, 5.0};
	int shape[] = {2};
	TensorHandle p = tensor_create(v, shape, 1, 1);
	param_register("b", p);
	cr_assert_eq(param_count(), 1);

	backend_release_all_persistent();

	cr_assert_eq(param_count(), 0);
}
#endif /* BACKEND_TAPE */

/* release_all_persistent skips NULL param slots without crashing (the
   `if (!t) continue;` guard at shims.c:65). An empty registry drives the loop
   zero times and exercises the param_clear + arena_free_all tail. */
Test(tape_shims, release_all_persistent_empty_registry) {
	param_clear();
	backend_release_all_persistent();
	cr_assert_eq(param_count(), 0);
}

/* ----------------------------------------------------------------------
   shims.c — live-count probes, backend_name, mlx_compile accessors.
   ---------------------------------------------------------------------- */

Test(tape_shims, live_count_probes) {
	param_clear();
	int live = tensor_live_count();      /* shims.c:76 — reads tape_size */
	int peak = tensor_peak_live_count(); /* shims.c:79 — reads g_tape_peak */
	cr_assert_geq(live, 0);
	cr_assert_geq(peak, live);
	param_clear();
}

#ifdef BACKEND_TAPE
Test(tape_shims, backend_name_is_tape) {
	cr_assert_str_eq(backend_name(), "tape"); /* shims.c:84 */
}
#endif /* BACKEND_TAPE */

/* mlx_compile accessors are tape no-ops (shims.c:90,93,95): disabled, zero
   invocations, reset is a no-op. */
Test(tape_shims, mlx_compile_accessors_are_noops) {
	cr_assert_eq(tensor_mlx_compile_enabled(), 0);
	cr_assert_eq(tensor_mlx_compile_invocations(), 0);
	tensor_mlx_compile_reset_stats(); /* no-op, must not crash */
	cr_assert_eq(tensor_mlx_compile_invocations(), 0);
}

/* ----------------------------------------------------------------------
   shims.c — tensor_print: rank-0 scalar arm + rank-N list arm.
   ---------------------------------------------------------------------- */

/* Rank-0 scalar arm (shims.c:100-101). Output goes to stdout; we only assert
   it runs without crashing. */
Test(tape_shims, print_scalar) {
	TensorHandle s = tensor_create_scalar(3.14, 0);
	tensor_print(s);
}

/* Rank-N arm (shims.c:103-108) — the `[a, b, c]` list dump, including the
   i>0 comma branch. */
Test(tape_shims, print_vector) {
	param_clear();
	double v[] = {1.0, 2.0, 3.0};
	int shape[] = {3};
	TensorHandle t = tensor_create(v, shape, 1, 0);
	tensor_print(t);
	param_clear();
}

/* ----------------------------------------------------------------------
   shared/training/ffi_shims.c — the *_return wrappers.
   ---------------------------------------------------------------------- */

/* tensor_backward_return (ffi_shims.c:21-23): runs backward, returns the same
   handle, and the grad must be populated. */
Test(tape_ffi_shims, backward_return_passthrough) {
	param_clear();
	double v[] = {2.0, 3.0};
	int shape[] = {2};
	TensorHandle p = tensor_create(v, shape, 1, 1);
	param_register("w", p);
	TensorHandle loss = tensor_sum(p);
	TensorHandle ret = tensor_backward_return(loss);
	cr_assert_eq(ret, loss);
	/* d(sum)/dx = 1 for each element. */
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12);
	param_clear();
}

/* param_register_return (ffi_shims.c:26-29): marks requires_grad and registers,
   returns the handle. */
Test(tape_ffi_shims, param_register_return_registers) {
	param_clear();
	double v[] = {1.0};
	int shape[] = {1};
	TensorHandle p = tensor_create(v, shape, 1, 0);
	TensorHandle ret = param_register_return("w", p);
	cr_assert_eq(ret, p);
	cr_assert_eq(param_count(), 1);
	cr_assert_eq(tensor_requires_grad(p), 1);
	param_clear();
}

/* param_register_buffer_return (ffi_shims.c:35-37): registers a buffer (no
   requires_grad flip), returns the handle. */
Test(tape_ffi_shims, param_register_buffer_return_registers) {
	param_clear();
	double v[] = {7.0};
	int shape[] = {1};
	TensorHandle p = tensor_create(v, shape, 1, 0);
	TensorHandle ret = param_register_buffer_return("buf", p);
	cr_assert_eq(ret, p);
	cr_assert_eq(param_count(), 1);
	param_clear();
}

/* param_zero_all_grads_return (ffi_shims.c:40-43): zeroes grads, returns 0. */
Test(tape_ffi_shims, zero_all_grads_return) {
	param_clear();
	double v[] = {2.0, 4.0};
	int shape[] = {2};
	TensorHandle p = tensor_create(v, shape, 1, 1);
	param_register("w", p);
	tensor_backward(tensor_sum(p));
	int rc = param_zero_all_grads_return(99);
	cr_assert_eq(rc, 0);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12);
	param_clear();
}

/* tensor_to_doubles_return (ffi_shims.c:46-48): flattens into the buffer and
   returns the same buffer pointer. */
Test(tape_ffi_shims, to_doubles_return_passthrough) {
	param_clear();
	double v[] = {1.5, 2.5, 3.5};
	int shape[] = {3};
	TensorHandle t = tensor_create(v, shape, 1, 0);
	double buf[3] = {0};
	double* ret = tensor_to_doubles_return(t, buf);
	cr_assert_eq(ret, buf);
	cr_assert_float_eq(buf[0], 1.5, 1e-12);
	cr_assert_float_eq(buf[2], 3.5, 1e-12);
	param_clear();
}

/* tensor_backward_return_loss (ffi_shims.c:51-53): requires_grad path — runs
   backward, returns the supplied loss value. */
Test(tape_ffi_shims, backward_return_loss_requires_grad) {
	param_clear();
	double v[] = {3.0, 5.0};
	int shape[] = {2};
	TensorHandle p = tensor_create(v, shape, 1, 1);
	param_register("w", p);
	TensorHandle loss = tensor_sum(p);
	double ret = tensor_backward_return_loss(loss, 8.0);
	cr_assert_float_eq(ret, 8.0, 1e-12);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12);
	param_clear();
}

/* tensor_backward_return_loss with a non-requires-grad tensor: skips backward
   (the false arm of the if at ffi_shims.c:52), still returns the loss value. */
Test(tape_ffi_shims, backward_return_loss_no_grad) {
	param_clear();
	double v[] = {3.0, 5.0};
	int shape[] = {2};
	TensorHandle t = tensor_create(v, shape, 1, 0); /* requires_grad = 0 */
	double ret = tensor_backward_return_loss(t, 2.0);
	cr_assert_float_eq(ret, 2.0, 1e-12);
	param_clear();
}

/* idrisml_seq (ffi_shims.c:56-58): evaluates a, returns b. */
Test(tape_ffi_shims, idrisml_seq_returns_second) {
	int a = 1;
	int b = 2;
	void* ret = idrisml_seq(&a, &b);
	cr_assert_eq(ret, &b);
}

/* backend_reset_for_eval_return (ffi_shims.c:61-64): runs reset, returns its
   dummy arg unchanged. */
Test(tape_ffi_shims, reset_for_eval_return) {
	param_clear();
	int rc = backend_reset_for_eval_return(42);
	cr_assert_eq(rc, 42);
	param_clear();
}

/* backend_profile_reset_return + backend_profile_report_return
   (ffi_shims.c:67-70, 73-76): both run the profile op and return the dummy. */
Test(tape_ffi_shims, profile_reset_and_report_return) {
	int r1 = backend_profile_reset_return(7);
	cr_assert_eq(r1, 7);
	int r2 = backend_profile_report_return(9);
	cr_assert_eq(r2, 9);
}

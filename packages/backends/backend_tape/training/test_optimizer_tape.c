/* Criterion suite — tape optimizer edge paths + param-registry load guards.
 *
 * Targets backend_tape/training/optimizer.c uncovered branches:
 *   - SKIP_LSTM_INIT skip for params ending in _h0 / _c0 (289-292)
 *   - RMSprop no-momentum update branch (324)
 *   - unknown opt->type default -> abort (348,350) via a SIGABRT death test
 *   - get_m / get_v before buffers allocated -> zero-fill (388-389, 399-400)
 *   - clip_grad_value_filtered (455-467) + clip_grad_norm_filtered scale
 *     branch (485-492), both driven through the public native_train_step
 *     clip-mode dispatch.
 * and shared/training/param_registry.c:
 *   - param_load_data size-mismatch guard (170-172)
 *   - param_load_data_int64 mismatch + success (177-185)
 *
 * Colocated under backend_tape/ but compiled into all three backend test
 * binaries via the test_*.c glob; every assert calls public renamed FFI so
 * the same oracle runs on tape/torch/mlx (the death test stays tape-driven
 * because the abort lives in tape's optimizer step).
 */

#include <criterion/criterion.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include "backend.h"
#include "test_helpers.h"

/* ----------------------------------------------------------------------
   RMSprop without momentum — momentum == 0 takes the else branch:
   w <- w - lr * g / (sqrt(v) + eps), with v = (1-alpha)*g^2 on step 1.
   ---------------------------------------------------------------------- */
Test(tape_optimizer_edge, rmsprop_no_momentum_step) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("rms_a", a);
	double lr = 0.1, alpha = 0.99, eps = 1e-8, wd = 0.0, momentum = 0.0;
	OptimizerHandle opt = optimizer_create_rmsprop(lr, alpha, eps, wd, momentum);
	/* loss = a*a => grad(a) = 2a = 2.0 */
	TensorHandle loss = tensor_mul(a, a);
	native_train_step(opt, 0, 0.0, loss, 0.0);
	/* v = (1-0.99)*4 = 0.04; avg = sqrt(0.04)+1e-8 = 0.2; step = 0.1*2/0.2 = 1.0
	   => a = 1.0 - 1.0 = 0.0 (within readback tol). */
	double avg = 0.2 + eps;
	double expect = 1.0 - lr * 2.0 / avg;
	cr_assert_float_eq(tensor_item(a), expect, TEST_TOL_RELAXED,
	                   "rmsprop no-momentum step (got %.9f, want %.9f)", tensor_item(a), expect);
	optimizer_free(opt);
	param_clear();
}

#ifdef BACKEND_TAPE
/* ----------------------------------------------------------------------
   SKIP_LSTM_INIT — params whose name ends in _h0 / _c0 are skipped by the
   step loop (the LSTM learned-initial-state diagnostic).
   ---------------------------------------------------------------------- */
Test(tape_optimizer_edge, skip_lstm_init_leaves_h0_c0) {
	setenv("SKIP_LSTM_INIT", "1", 1);
	param_clear();
	TensorHandle h0 = tensor_create_scalar(1.0, 1);
	TensorHandle c0 = tensor_create_scalar(1.0, 1);
	TensorHandle w = tensor_create_scalar(1.0, 1);
	param_register("cell_h0", h0);
	param_register("cell_c0", c0);
	param_register("cell_w", w);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	/* loss = h0*h0 + c0*c0 + w*w => grad = 2 each */
	TensorHandle loss =
	    tensor_add(tensor_add(tensor_mul(h0, h0), tensor_mul(c0, c0)), tensor_mul(w, w));
	native_train_step(opt, 0, 0.0, loss, 0.0);
	cr_assert_float_eq(tensor_item(h0), 1.0, TEST_TOL_RELAXED, "_h0 skipped (got %.9f)",
	                   tensor_item(h0));
	cr_assert_float_eq(tensor_item(c0), 1.0, TEST_TOL_RELAXED, "_c0 skipped (got %.9f)",
	                   tensor_item(c0));
	/* cell_w not skipped: 1 - 0.1*2 = 0.8 */
	cr_assert_float_eq(tensor_item(w), 0.8, TEST_TOL_RELAXED, "cell_w stepped (got %.9f)",
	                   tensor_item(w));
	optimizer_free(opt);
	param_clear();
	unsetenv("SKIP_LSTM_INIT");
}
#endif /* BACKEND_TAPE */

#ifdef BACKEND_TAPE
/* ----------------------------------------------------------------------
   Unknown opt->type -> default branch fprintf + abort(). Set the type via
   the serialization meta vector (slot 0) to an out-of-range value, register
   a param with a grad so the inner switch is reached, and step.
   abort() flushes no gcov in the forked child, so the abort body is
   GCOVR_EXCL-wrapped in optimizer.c (named for this test).
   ---------------------------------------------------------------------- */
Test(tape_optimizer_edge, unknown_type_aborts, .signal = SIGABRT) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("bad_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	/* meta: [type, lr, b1, b2, eps, alpha, wd, momentum, t] */
	double meta[9] = {99.0, 0.1, 0.9, 0.999, 1e-8, 0.99, 0.0, 0.0, 0.0};
	optimizer_set_meta(opt, meta);
	TensorHandle loss = tensor_mul(a, a);
	native_train_step(opt, 0, 0.0, loss, 0.0); /* hits switch default -> abort */
	                                           /* unreachable */
}
#endif /* BACKEND_TAPE */

/* ----------------------------------------------------------------------
   get_m / get_v before buffers are allocated -> zero-fill the out buffer.
   No step has run, so opt->allocated == 0.
   ---------------------------------------------------------------------- */
Test(tape_optimizer_edge, get_m_before_alloc_is_zero) {
	param_clear();
	TensorHandle a = tensor_create_scalar(7.0, 1);
	param_register("m_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double out = 1234.0;
	optimizer_get_m(opt, 0, &out);
	cr_assert_float_eq(out, 0.0, TEST_TOL_TIGHT, "get_m pre-alloc zero-fills (got %.9f)", out);
	optimizer_free(opt);
	param_clear();
}

Test(tape_optimizer_edge, get_v_before_alloc_is_zero) {
	param_clear();
	TensorHandle a = tensor_create_scalar(7.0, 1);
	param_register("v_a", a);
	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);
	double out = 4321.0;
	optimizer_get_v(opt, 0, &out);
	cr_assert_float_eq(out, 0.0, TEST_TOL_TIGHT, "get_v pre-alloc zero-fills (got %.9f)", out);
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   clip_grad_value_filtered (clip_mode == 1). Single-optimizer (empty owned
   set) clips every registered param. grad(a) = 2*3 = 6 -> clamped to 2.0,
   then SGD lr=0.1 steps: a = 3 - 0.1*2 = 2.8.
   ---------------------------------------------------------------------- */
Test(tape_optimizer_edge, clip_grad_value_filtered_clamps) {
	param_clear();
	TensorHandle a = tensor_create_scalar(3.0, 1);
	param_register("clipv_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a);      /* grad = 6 */
	native_train_step(opt, 1, 2.0, loss, 0.0); /* clip_mode 1, max_val 2.0 */
	cr_assert_float_eq(tensor_item(a), 2.8, TEST_TOL_RELAXED,
	                   "grad clamped to 2.0 then SGD step -> 2.8 (got %.9f)", tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

/* clip value with a grad already below max_val: no clamp, full grad steps. */
Test(tape_optimizer_edge, clip_grad_value_filtered_no_clamp) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("clipv_b", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 2, < max_val 10 */
	native_train_step(opt, 1, 10.0, loss, 0.0);
	cr_assert_float_eq(tensor_item(a), 0.8, TEST_TOL_RELAXED,
	                   "grad 2 unclamped -> SGD step 0.8 (got %.9f)", tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

#ifdef BACKEND_TAPE
/* ----------------------------------------------------------------------
   clip_grad_norm_filtered (clip_mode == 2), norm > max_norm -> scale branch.
   grad(a) = 2*4 = 8 -> norm 8 > max_norm 2 -> scale 2/8 = 0.25 -> grad 2.0,
   then SGD lr=0.1: a = 4 - 0.1*2 = 3.8.
   ---------------------------------------------------------------------- */
Test(tape_optimizer_edge, clip_grad_norm_filtered_rescales) {
	param_clear();
	TensorHandle a = tensor_create_scalar(4.0, 1);
	param_register("clipn_a", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a);      /* grad = 8 */
	native_train_step(opt, 2, 2.0, loss, 0.0); /* clip_mode 2, max_norm 2.0 */
	cr_assert_float_eq(tensor_item(a), 3.8, TEST_TOL_RELAXED,
	                   "grad-norm 8 rescaled to 2 then SGD step -> 3.8 (got %.9f)", tensor_item(a));
	optimizer_free(opt);
	param_clear();
}
#endif /* BACKEND_TAPE */

/* norm <= max_norm: no rescale, full grad steps (norm-clip below threshold). */
Test(tape_optimizer_edge, clip_grad_norm_filtered_no_rescale) {
	param_clear();
	TensorHandle a = tensor_create_scalar(1.0, 1);
	param_register("clipn_b", a);
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	TensorHandle loss = tensor_mul(a, a); /* grad = 2, norm 2 <= max 10 */
	native_train_step(opt, 2, 10.0, loss, 0.0);
	cr_assert_float_eq(tensor_item(a), 0.8, TEST_TOL_RELAXED,
	                   "grad-norm 2 under threshold -> SGD step 0.8 (got %.9f)", tensor_item(a));
	optimizer_free(opt);
	param_clear();
}

/* ----------------------------------------------------------------------
   param_load_data size-mismatch guard (170-172): wrong numel -> early
   return, destination tensor unchanged.
   ---------------------------------------------------------------------- */
Test(tape_param_registry_edge, load_data_size_mismatch_is_noop) {
	param_clear();
	TensorHandle a = tensor_create_scalar(5.0, 0); /* numel 1 */
	param_register("ld_a", a);
	double data[3] = {1.0, 2.0, 3.0};
	param_load_data(0, data, 3); /* dest numel 1 != 3 -> guard fires, no write */
	cr_assert_float_eq(tensor_item(a), 5.0, TEST_TOL_TIGHT,
	                   "size mismatch left tensor unchanged (got %.9f)", tensor_item(a));
	param_clear();
}

/* param_load_data success: matching numel writes the new value. */
Test(tape_param_registry_edge, load_data_match_writes) {
	param_clear();
	TensorHandle a = tensor_create_scalar(5.0, 0);
	param_register("ld_b", a);
	double data[1] = {9.0};
	param_load_data(0, data, 1);
	cr_assert_float_eq(tensor_item(a), 9.0, TEST_TOL_TIGHT, "matching numel wrote 9.0 (got %.9f)",
	                   tensor_item(a));
	param_clear();
}

/* ----------------------------------------------------------------------
   param_load_data_int64 (177-185): mismatch guard + matching-numel write.
   The int64 buffer is narrowed by the port's load_int64.
   ---------------------------------------------------------------------- */
Test(tape_param_registry_edge, load_data_int64_size_mismatch_is_noop) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 0); /* numel 1 */
	param_register("ld64_a", a);
	int64_t data[2] = {10, 20};
	param_load_data_int64(0, data, 2); /* dest numel 1 != 2 -> guard fires */
	cr_assert_float_eq(tensor_item(a), 2.0, TEST_TOL_TIGHT,
	                   "int64 size mismatch left tensor unchanged (got %.9f)", tensor_item(a));
	param_clear();
}

Test(tape_param_registry_edge, load_data_int64_match_writes) {
	param_clear();
	TensorHandle a = tensor_create_scalar(2.0, 0);
	param_register("ld64_b", a);
	int64_t data[1] = {42};
	param_load_data_int64(0, data, 1);
	cr_assert_float_eq(tensor_item(a), 42.0, TEST_TOL_RELAXED,
	                   "int64 matching numel wrote 42 (got %.9f)", tensor_item(a));
	param_clear();
}

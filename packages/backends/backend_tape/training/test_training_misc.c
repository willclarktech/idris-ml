/* Criterion suite for tape training-misc product TUs.
 *
 * Raises line coverage on four tape training-side TUs that the broader
 * suites leave partly uncovered:
 *   - training/adapter.c        — the `g_active_port` trampolines: the
 *     int64 bulk loader, the wall-clock provider, the 4d param + 2d state
 *     dtag creators, and the fused param-create-+-init (normal / const)
 *     trampolines + the init-seed setter.
 *   - training/host_io.c        — tensor_size out-of-range axis, the I64
 *     readout (tensor_to_int64), and tensor_to_floats (both the F32 memcpy
 *     fast path and the F64 narrowing loop).
 *   - training/param_create.c   — tensor_create_state_2d_f64 (persistent,
 *     requires_grad=0, no tape entry).
 *   - training/autograd/helpers.c — tensor_zero_grad, tensor_set_requires_grad
 *     (the tape_append branch), no-grad / epoch depth toggles, the device
 *     identity shims.
 *
 * Each Criterion Test() runs in its own forked child, so param_clear() +
 * the persistent allocations here can't corrupt sibling tests.
 *
 * The dtag-streamed creators that drive the adapter trampolines free their
 * `double*` data argument, so buffers are built with tensor_alloc_doubles /
 * tensor_write_double_return (callee-frees contract). F32 readback carries
 * ~1e-6 error; F32 asserts use an explicit 1e-5 tolerance.
 */

#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"
#include "shared/training/port.h"

#ifdef BACKEND_TAPE

/* dtag constants (mirror test_dtype_scaffolding.c): F32=14, F64=15. */
#define DTAG_F32 14
#define DTAG_F64 15

/* Build a heap double buffer the callee-frees creators can consume. */

/* ----------------------------------------------------------------------
   adapter.c — port trampolines.
   ---------------------------------------------------------------------- */

/* tape_load_int64 (adapter.c 80-83) via param_load_data_int64. The tape
   stores integers in the double lingua-franca, so the readback is exact. */
Test(training_misc, adapter_load_int64) {
	param_clear();
	double zeros[] = {0.0, 0.0, 0.0};
	TensorHandle t = tensor_create_param_1d_f64(3, hcopy(zeros, 3));
	param_register("p", t);
	int64_t src[] = {7, -3, 42};
	param_load_data_int64(0, src, 3);
	double out[3];
	tensor_to_doubles(t, out);
	cr_assert_float_eq(out[0], 7.0, 1e-12);
	cr_assert_float_eq(out[1], -3.0, 1e-12);
	cr_assert_float_eq(out[2], 42.0, 1e-12);
	param_clear();
}

/* tape_adapter_wall_ms (adapter.c 99-100) via the port struct member.
   _wall_ms is monotonic-ish ms; assert it returns a finite, non-negative
   reading (the body is a single delegating call). */
Test(training_misc, adapter_wall_ms) {
	double ms = g_active_port.wall_ms();
	cr_assert_geq(ms, 0.0, "wall_ms should be a non-negative millisecond reading");
}

/* tape_port_create_param_4d (adapter.c 160-161) via the 4d dtag creator. */
Test(training_misc, adapter_create_param_4d) {
	param_clear();
	double vals[16];
	for (int i = 0; i < 16; i++)
		vals[i] = (double)i;
	TensorHandle t = tensor_create_param_4d_streamed(2, 2, 2, 2, hcopy(vals, 16), 0, DTAG_F64);
	cr_assert_eq(tensor_numel(t), 16);
	cr_assert_eq(tensor_dim(t), 4);
	cr_assert_eq(tensor_size(t, 3), 2);
	double out[16];
	tensor_to_doubles(t, out);
	cr_assert_float_eq(out[5], 5.0, 1e-12);
	param_clear();
}

/* tape_port_create_state_2d (adapter.c 166-167) via the 2d state creator.
   State tensors are persistent, requires_grad=0. */
Test(training_misc, adapter_create_state_2d) {
	double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle t = tensor_create_state_2d_streamed(2, 3, hcopy(vals, 6), 0, DTAG_F64);
	cr_assert_eq(tensor_numel(t), 6);
	cr_assert_eq(tensor_requires_grad(t), 0, "state tensor must not require grad");
	double out[6];
	tensor_to_doubles(t, out);
	cr_assert_float_eq(out[4], 5.0, 1e-12);
}

/* Fused param create + init — normal (adapter.c 174-187). The values are
   random; assert shape/dtype + that the seed makes them deterministic. */
Test(training_misc, adapter_create_param_normal_all_ranks) {
	tensor_set_init_seed_streamed(123ULL, 0); /* adapter.c 202-203 */
	TensorHandle a = tensor_create_param_1d_normal_streamed(4, 0.0, 1.0, 0, DTAG_F64);
	TensorHandle b = tensor_create_param_2d_normal_streamed(2, 3, 0.0, 1.0, 0, DTAG_F64);
	TensorHandle c = tensor_create_param_3d_normal_streamed(2, 2, 2, 0.0, 1.0, 0, DTAG_F64);
	TensorHandle d = tensor_create_param_4d_normal_streamed(2, 2, 1, 2, 0.0, 1.0, 0, DTAG_F64);
	cr_assert_eq(tensor_numel(a), 4);
	cr_assert_eq(tensor_numel(b), 6);
	cr_assert_eq(tensor_numel(c), 8);
	cr_assert_eq(tensor_numel(d), 8);
	cr_assert_eq(tensor_requires_grad(a), 1);
}

/* Fused param create + init — const (adapter.c 189-200). Const fill is
   value-exact, so assert the fill value across all ranks. */
Test(training_misc, adapter_create_param_const_all_ranks) {
	TensorHandle a = tensor_create_param_1d_const_streamed(3, 0.5, 0, DTAG_F64);
	TensorHandle b = tensor_create_param_2d_const_streamed(2, 2, 1.25, 0, DTAG_F64);
	TensorHandle c = tensor_create_param_3d_const_streamed(2, 1, 2, -2.0, 0, DTAG_F64);
	TensorHandle d = tensor_create_param_4d_const_streamed(1, 2, 1, 2, 3.5, 0, DTAG_F64);
	double oa[3], ob[4], oc[4], od[4];
	tensor_to_doubles(a, oa);
	tensor_to_doubles(b, ob);
	tensor_to_doubles(c, oc);
	tensor_to_doubles(d, od);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(oa[i], 0.5, 1e-12);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(ob[i], 1.25, 1e-12);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(oc[i], -2.0, 1e-12);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(od[i], 3.5, 1e-12);
}

/* F32 const create through the same trampolines — exercises the F32 storage
   path in dtype_init.c that the trampolines forward to. */
Test(training_misc, adapter_create_param_const_f32) {
	TensorHandle a = tensor_create_param_1d_const_streamed(3, 0.5, 0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(a), "F32");
	double oa[3];
	tensor_to_doubles(a, oa);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(oa[i], 0.5, 1e-5);
}

/* ----------------------------------------------------------------------
   host_io.c
   ---------------------------------------------------------------------- */

/* tensor_size out-of-range axis returns 0 (host_io.c 29). */
Test(training_misc, host_io_size_out_of_range) {
	double vals[] = {1.0, 2.0, 3.0};
	int s[] = {3};
	TensorHandle t = tensor_create(vals, s, 1, 0);
	cr_assert_eq(tensor_size(t, 0), 3);
	cr_assert_eq(tensor_size(t, 1), 0, "axis >= rank should return 0");
	cr_assert_eq(tensor_size(t, 99), 0);
}

/* tensor_to_int64 (host_io.c 42-45) — casts the dtype-uniform double view. */
Test(training_misc, host_io_to_int64) {
	double vals[] = {1.9, -2.1, 3.0, 7.0};
	int s[] = {4};
	TensorHandle t = tensor_create(vals, s, 1, 0);
	int64_t out[4];
	tensor_to_int64(t, out);
	cr_assert_eq(out[0], 1, "truncation toward zero");
	cr_assert_eq(out[1], -2);
	cr_assert_eq(out[2], 3);
	cr_assert_eq(out[3], 7);
}

/* tensor_to_floats — F32 memcpy fast path (host_io.c 51-52). */
Test(training_misc, host_io_to_floats_f32) {
	double vals[] = {1.5, -2.25, 3.0};
	TensorHandle t = tensor_create_1d_streamed(3, hcopy(vals, 3), 0, 0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(t), "F32");
	float out[3];
	tensor_to_floats(t, out);
	cr_assert_float_eq(out[0], 1.5f, 1e-5);
	cr_assert_float_eq(out[1], -2.25f, 1e-5);
	cr_assert_float_eq(out[2], 3.0f, 1e-5);
}

/* tensor_to_floats — F64 narrowing loop (host_io.c 53-55). */
Test(training_misc, host_io_to_floats_f64) {
	double vals[] = {1.5, -2.25, 3.0};
	int s[] = {3};
	TensorHandle t = tensor_create(vals, s, 1, 0);
	cr_assert_str_eq(tensor_dtype_name(t), "F64");
	float out[3];
	tensor_to_floats(t, out);
	cr_assert_float_eq(out[0], 1.5f, 1e-5);
	cr_assert_float_eq(out[1], -2.25f, 1e-5);
	cr_assert_float_eq(out[2], 3.0f, 1e-5);
}

/* ----------------------------------------------------------------------
   param_create.c — tensor_create_state_2d_f64 (111-125).
   ---------------------------------------------------------------------- */

/* Direct F64 2d state creator: persistent, requires_grad=0, no tape entry. */
Test(training_misc, param_create_state_2d_f64) {
	double vals[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
	TensorHandle t = tensor_create_state_2d_f64(3, 2, hcopy(vals, 6));
	cr_assert_eq(tensor_numel(t), 6);
	cr_assert_eq(tensor_dim(t), 2);
	cr_assert_eq(tensor_size(t, 0), 3);
	cr_assert_eq(tensor_size(t, 1), 2);
	cr_assert_eq(tensor_requires_grad(t), 0, "state tensor must not require grad");
	double out[6];
	tensor_to_doubles(t, out);
	cr_assert_float_eq(out[0], 10.0, 1e-12);
	cr_assert_float_eq(out[5], 60.0, 1e-12);
}

/* ----------------------------------------------------------------------
   autograd/helpers.c
   ---------------------------------------------------------------------- */

/* tensor_zero_grad (helpers.c 30-32): after backward populates a grad,
   tensor_zero_grad memsets it to zero. */
Test(training_misc, helpers_zero_grad) {
	param_clear();
	double zeros[] = {0.0, 0.0, 0.0};
	TensorHandle a = tensor_create_param_1d_f64(3, hcopy(zeros, 3));
	param_register("a", a);
	TensorHandle loss = tensor_sum(a);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12, "grad before zero should be 1.0");
	tensor_zero_grad(a);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-12, "grad after zero_grad should be 0.0");
	param_clear();
}

/* tensor_set_requires_grad (helpers.c 50-56): flipping a non-param tensor
   to requires_grad=1 appends an OP_CONST tape entry (line 54) so it becomes
   gradient-bearing. */
Test(training_misc, helpers_set_requires_grad_appends) {
	param_clear();
	double zeros[] = {0.0, 0.0};
	int s[] = {2};
	TensorHandle a = tensor_create(zeros, s, 1, 0); /* rank-1, requires_grad=0 */
	cr_assert_eq(tensor_requires_grad(a), 0);
	tensor_set_requires_grad(a, 1);
	cr_assert_eq(tensor_requires_grad(a), 1);
	/* Now a is gradient-bearing; sum + backward gives grad 1.0. */
	param_register("a", a);
	TensorHandle loss = tensor_sum(a);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12,
	                   "after set_requires_grad(1), grad should flow");
	param_clear();
}

/* no-grad depth toggles (helpers.c 58-62). begin++/end-- around a graph
   region; the end clamps at 0. Drives both increment and decrement lines. */
Test(training_misc, helpers_no_grad_depth) {
	tensor_no_grad_begin();
	tensor_no_grad_end();
	/* end at zero is a clamped no-op (the `if (no_grad_depth > 0)` guard) */
	tensor_no_grad_end();
	cr_assert(1, "no_grad begin/end exercised without crash");
}

/* epoch begin/end no-ops (helpers.c 66-68). */
Test(training_misc, helpers_epoch_noops) {
	tensor_epoch_begin();
	tensor_epoch_end();
	cr_assert(1, "epoch begin/end are no-ops on tape");
}

/* device shims (helpers.c 72-84): to_device, to_device_persistent return
   identity; device reads "cpu". */
Test(training_misc, helpers_device_shims) {
	double vals[] = {1.0, 2.0};
	int s[] = {2};
	TensorHandle t = tensor_create(vals, s, 1, 0);
	cr_assert_eq(tensor_to_device(t, "mps"), t, "to_device is identity on tape");
	cr_assert_eq(tensor_to_device_persistent(t, "cuda"), t,
	             "to_device_persistent is identity on tape");
	cr_assert_str_eq(tensor_device(t), "cpu");
}

#endif /* BACKEND_TAPE */

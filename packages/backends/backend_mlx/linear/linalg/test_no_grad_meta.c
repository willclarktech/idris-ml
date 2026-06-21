/* mlx-only Criterion suite for meta-writing ops under no-grad.
 *
 * tape_append (backend_mlx/tape.cpp) returns -1 while
 * no_grad_depth_mlx > 0, but a meta-carrying call site that enters
 * its `if (requires_grad)` block anyway (param inputs keep
 * requires_grad=true inside withNoGrad) must not do
 * `tape[idx].meta = ...` unguarded: with an empty tape that is a
 * write through ((TapeEntry*)NULL)[-1] (deterministic SIGSEGV —
 * what these tests catch, since Criterion runs each test in a fresh
 * forked process with an empty tape); with a non-empty tape it is a
 * heap buffer underrun that trashes the word before the tape
 * vector's allocation (the layout-dependent SIGABRT / "invalid
 * memory reference" that killed example-a2c [mlx], example-dqn
 * [mlx-gpu] and example-mountain-car [mlx] in CI run 27373449876).
 *
 * Each test drives one representative meta-carrying op through a
 * no-grad forward with a grad-requiring input, mimicking the Idris
 * FFI contract: returned handles are retained (wrap-and-retain) and
 * results are read back INSIDE the bracket — tensor_no_grad_end runs
 * the generation sweep, which by design reclaims rc==1 block-local
 * handles, so post-bracket reads are out of contract. The numerics
 * must be unaffected by grad mode.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

Test(mlx_no_grad_meta, linear_2d_under_no_grad_with_param) {
	/* W: [2,2] param (requires_grad survives no-grad), X: [1,2]. */
	double wd[] = {1.0, 2.0, 3.0, 4.0};
	double xd[] = {10.0, 100.0};
	TensorHandle W = mk2d(2, 2, wd, /*requires_grad=*/1);
	TensorHandle X = mk2d(1, 2, xd, /*requires_grad=*/0);

	tensor_no_grad_begin();
	TensorHandle Y = tensor_linear_2d(W, X, NULL); /* Y = X @ W^T */
	tensor_retain_handle(Y);                       /* FFI wrap-and-retain */
	double out[2] = {0.0, 0.0};
	tensor_to_doubles(Y, out);
	tensor_no_grad_end();

	cr_assert_float_eq(out[0], 210.0, 1e-9, "Y[0] expected 210 got %f", out[0]);
	cr_assert_float_eq(out[1], 430.0, 1e-9, "Y[1] expected 430 got %f", out[1]);
}

Test(mlx_no_grad_meta, sum_dim_under_no_grad_with_param) {
	double td[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle t = mk2d(2, 3, td, /*requires_grad=*/1);

	tensor_no_grad_begin();
	TensorHandle s = tensor_sum_dim(t, /*dim=*/1, /*keepdim=*/0);
	tensor_retain_handle(s);
	double out[2] = {0.0, 0.0};
	tensor_to_doubles(s, out);
	tensor_no_grad_end();

	cr_assert_float_eq(out[0], 6.0, 1e-9, "row0 sum expected 6 got %f", out[0]);
	cr_assert_float_eq(out[1], 15.0, 1e-9, "row1 sum expected 15 got %f", out[1]);
}

Test(mlx_no_grad_meta, stack_under_no_grad_with_param) {
	double ad[] = {1.0, 2.0};
	double bd[] = {3.0, 4.0};
	TensorHandle a = tensor_create_1d_f64(2, hcopy(ad, 2), /*requires_grad=*/1);
	TensorHandle b = tensor_create_1d_f64(2, hcopy(bd, 2), /*requires_grad=*/0);
	TensorHandle parts[2] = {a, b};

	tensor_no_grad_begin();
	TensorHandle s = tensor_stack(parts, 2, /*dim=*/0);
	tensor_retain_handle(s);
	double out[4] = {0.0, 0.0, 0.0, 0.0};
	tensor_to_doubles(s, out);
	tensor_no_grad_end();

	cr_assert_float_eq(out[0], 1.0, 1e-9, "stack[0] expected 1 got %f", out[0]);
	cr_assert_float_eq(out[3], 4.0, 1e-9, "stack[3] expected 4 got %f", out[3]);
}

#endif /* BACKEND_MLX */

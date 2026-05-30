/* Test suite for the mx::compile integration in the MLX backend —
 * Criterion port.
 *
 * Carried over from the previous standalone main() program. The 7 helper
 * functions become 7 Test() cases under the `mlx_compile` suite. Each
 * Test() forks a fresh process, so cross-test env-var leakage is
 * automatically contained — but every case still leads with the
 * appropriate setenv/unsetenv to make state-resets explicit, matching
 * the original helper bodies.
 *
 * The whole file is wrapped in #ifdef BACKEND_MLX. tensor_mlx_compile_*
 * symbols are mlx-only (no-op stubs on tape/torch); gating at the
 * preprocessor level matches the original Make-recipe-only gate.
 */
#ifdef BACKEND_MLX

#include <criterion/criterion.h>
#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT_EQ(msg, got, expected) do { \
    long _aeq_g = (long)(got); \
    long _aeq_e = (long)(expected); \
    cr_assert_eq(_aeq_g, _aeq_e, \
        "%s: got %ld expected %ld", msg, _aeq_g, _aeq_e); \
} while (0)

#define ASSERT_NEAR(msg, got, expected, tol) do { \
    double _an_g = (got); \
    double _an_e = (expected); \
    cr_assert_float_eq(_an_g, _an_e, (tol), \
        "%s: got %.6f expected %.6f (tol %.6e)", msg, _an_g, _an_e, (double)(tol)); \
} while (0)

#define ASSERT_TRUE(msg, cond) \
    cr_assert((cond), "%s", msg)


/* ================================================================
   Stage 1: env var infrastructure
   ================================================================ */

Test(mlx_compile, disabled_by_default) {
    unsetenv("MLX_COMPILE");
    ASSERT_EQ("MLX_COMPILE unset -> disabled", tensor_mlx_compile_enabled(), 0);
}

Test(mlx_compile, enabled_via_env) {
    setenv("MLX_COMPILE", "1", 1);
    ASSERT_EQ("MLX_COMPILE=1 -> enabled", tensor_mlx_compile_enabled(), 1);
    unsetenv("MLX_COMPILE");
}

Test(mlx_compile, explicit_disable) {
    setenv("MLX_COMPILE", "0", 1);
    ASSERT_EQ("MLX_COMPILE=0 -> disabled", tensor_mlx_compile_enabled(), 0);
    unsetenv("MLX_COMPILE");
}

/* ================================================================
   Stage 2: compile-path probe + branch in tensor_backward

   tensor_mlx_compile_invocations() counts how many times the
   compile-enabled code path has been entered. The eager path leaves
   it at 0. tensor_mlx_compile_reset_stats() zeros the counter for
   test isolation.

   At this stage, the "compile path" is a no-op wrapper around the
   eager path — only the counter increments. Real mx::compile wiring
   lands in Stage 3.
   ================================================================ */

Test(mlx_compile, not_invoked_when_disabled) {
    unsetenv("MLX_COMPILE");
    tensor_mlx_compile_reset_stats();
    param_clear();

    TensorHandle w = tensor_create_scalar(3.0, 1);
    param_register("w", w);
    TensorHandle x = tensor_create_scalar(2.0, 0);
    TensorHandle y = tensor_mul(w, x);

    tensor_backward(y);

    ASSERT_EQ("invocations stays 0 (disabled)", tensor_mlx_compile_invocations(), 0);
    ASSERT_NEAR("grad w = x = 2.0 (eager)", param_grad_item(0), 2.0, 1e-6);

    tensor_free(w); tensor_free(x); tensor_free(y);
    param_clear();
}

Test(mlx_compile, invoked_when_enabled) {
    setenv("MLX_COMPILE", "1", 1);
    tensor_mlx_compile_reset_stats();
    param_clear();

    TensorHandle w = tensor_create_scalar(3.0, 1);
    param_register("w", w);
    TensorHandle x = tensor_create_scalar(2.0, 0);
    TensorHandle y = tensor_mul(w, x);

    tensor_backward(y);

    ASSERT_TRUE("invocations > 0 (enabled)", tensor_mlx_compile_invocations() > 0);
    ASSERT_NEAR("grad w = x = 2.0 (compile)", param_grad_item(0), 2.0, 1e-6);

    tensor_free(w); tensor_free(x); tensor_free(y);
    param_clear();
    unsetenv("MLX_COMPILE");
}

/* ================================================================
   Stage 3: gradient parity between eager and compile paths

   The compile branch must produce gradients ULP-close to the eager
   branch on any forward graph. This stage refactors the compile-branch
   forward closure to take constants (non-param tape inputs) as
   explicit function arguments instead of captured state, so they
   won't get baked into the compiled graph at trace time.

   At Stage 3 the compile branch still calls eager mx::vjp; the
   refactor is observable only via the parity test. Real mx::compile
   wiring lands in Stage 4.
   ================================================================ */

/* Helper: run y = sigmoid(w*x + b)^2 backward through current settings,
   returning grads in *gw, *gb. */
static void run_simple_backward(double w_val, double x_val, double b_val,
                                double* gw, double* gb) {
    param_clear();
    TensorHandle w = tensor_create_scalar(w_val, 1);
    TensorHandle b = tensor_create_scalar(b_val, 1);
    param_register("w", w);
    param_register("b", b);

    TensorHandle x = tensor_create_scalar(x_val, 0);
    TensorHandle wx = tensor_mul(w, x);
    TensorHandle z  = tensor_add(wx, b);
    TensorHandle s  = tensor_sigmoid(z);
    TensorHandle y  = tensor_mul(s, s);
    tensor_backward(y);

    *gw = param_grad_item(0);
    *gb = param_grad_item(1);

    tensor_free(w); tensor_free(b); tensor_free(x);
    tensor_free(wx); tensor_free(z); tensor_free(s); tensor_free(y);
    param_clear();
}

Test(mlx_compile, grad_parity_simple) {
    double w = 0.7, x = 1.3, b = -0.2;

    /* Eager (MLX_COMPILE=0) baseline */
    unsetenv("MLX_COMPILE");
    tensor_mlx_compile_reset_stats();
    double gw_eager, gb_eager;
    run_simple_backward(w, x, b, &gw_eager, &gb_eager);
    ASSERT_EQ("eager: invocations=0", tensor_mlx_compile_invocations(), 0);

    /* Compile (MLX_COMPILE=1) */
    setenv("MLX_COMPILE", "1", 1);
    tensor_mlx_compile_reset_stats();
    double gw_compile, gb_compile;
    run_simple_backward(w, x, b, &gw_compile, &gb_compile);
    ASSERT_TRUE("compile: invocations > 0", tensor_mlx_compile_invocations() > 0);

    /* Parity: f32 ULP tolerance. mlx is float32 internal. */
    ASSERT_NEAR("grad w parity", gw_compile, gw_eager, 1e-5);
    ASSERT_NEAR("grad b parity", gb_compile, gb_eager, 1e-5);

    unsetenv("MLX_COMPILE");
}

/* Same as above but exercises a graph where the input value (a
   "constant" in our tape's vocabulary) appears at multiple points.
   This is the case that would silently break if the compile branch
   were to bake constants into the graph at trace time and reuse them
   across calls with different input values. */
Test(mlx_compile, grad_parity_with_changing_constants) {
    /* Two backward passes with different input x values; both modes
       should produce the same gradients for each run, and the second
       run must use the new x (not the first run's x cached into a
       compiled graph). */
    double gw_eager_a, gb_eager_a, gw_eager_c, gb_eager_c;
    double gw_comp_a,  gb_comp_a,  gw_comp_c,  gb_comp_c;

    unsetenv("MLX_COMPILE");
    run_simple_backward(0.5, 1.0, 0.1, &gw_eager_a, &gb_eager_a);
    run_simple_backward(0.5, 2.0, 0.1, &gw_eager_c, &gb_eager_c);

    setenv("MLX_COMPILE", "1", 1);
    tensor_mlx_compile_reset_stats();
    run_simple_backward(0.5, 1.0, 0.1, &gw_comp_a, &gb_comp_a);
    run_simple_backward(0.5, 2.0, 0.1, &gw_comp_c, &gb_comp_c);

    ASSERT_NEAR("run-A grad w parity",  gw_comp_a, gw_eager_a, 1e-5);
    ASSERT_NEAR("run-A grad b parity",  gb_comp_a, gb_eager_a, 1e-5);
    ASSERT_NEAR("run-C grad w parity",  gw_comp_c, gw_eager_c, 1e-5);
    ASSERT_NEAR("run-C grad b parity",  gb_comp_c, gb_eager_c, 1e-5);

    /* Sanity: the two runs differ (x changed, so grads should too) */
    ASSERT_TRUE("run-A vs run-C grad w differs", fabs(gw_comp_a - gw_comp_c) > 1e-4);

    unsetenv("MLX_COMPILE");
}

#endif /* BACKEND_MLX */

/* Test suite for the mx::compile integration in the MLX backend.
   Job 3 Phase B — TDD driver.

   Build only when BACKEND=mlx (the probes are no-ops in tape/torch).
   Run: make test-mlx-compile */

#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;

#define ASSERT_EQ(msg, got, expected) do { \
    long _g = (long)(got), _e = (long)(expected); \
    if (_g != _e) { \
        printf("FAIL: %s: got %ld, expected %ld\n", msg, _g, _e); \
        failures++; \
    } else { \
        printf("ok: %s = %ld\n", msg, _g); \
    } \
} while(0)

#define ASSERT_NEAR(msg, got, expected, tol) do { \
    double _g = (got), _e = (expected); \
    if (fabs(_g - _e) > (tol)) { \
        printf("FAIL: %s: got %.6f, expected %.6f (tol %.6e)\n", msg, _g, _e, (double)(tol)); \
        failures++; \
    } else { \
        printf("ok: %s = %.6f\n", msg, _g); \
    } \
} while(0)

#define ASSERT_TRUE(msg, cond) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        failures++; \
    } else { \
        printf("ok: %s\n", msg); \
    } \
} while(0)

/* ================================================================
   Stage 1: env var infrastructure
   ================================================================ */

static void test_compile_disabled_by_default(void) {
    printf("\n--- compile env: disabled by default ---\n");
    unsetenv("MLX_COMPILE");
    ASSERT_EQ("MLX_COMPILE unset -> disabled", tensor_mlx_compile_enabled(), 0);
}

static void test_compile_enabled_via_env(void) {
    printf("\n--- compile env: MLX_COMPILE=1 enables ---\n");
    setenv("MLX_COMPILE", "1", 1);
    ASSERT_EQ("MLX_COMPILE=1 -> enabled", tensor_mlx_compile_enabled(), 1);
    unsetenv("MLX_COMPILE");
}

static void test_compile_explicit_disable(void) {
    printf("\n--- compile env: MLX_COMPILE=0 disables ---\n");
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

static void test_compile_not_invoked_when_disabled(void) {
    printf("\n--- compile NOT invoked when disabled ---\n");
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

static void test_compile_invoked_when_enabled(void) {
    printf("\n--- compile invoked when enabled ---\n");
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
   main
   ================================================================ */

int main(void) {
    setbuf(stdout, NULL);

    /* Stage 1: env var infrastructure */
    test_compile_disabled_by_default();
    test_compile_enabled_via_env();
    test_compile_explicit_disable();

    /* Stage 2: compile-path probe + branch */
    test_compile_not_invoked_when_disabled();
    test_compile_invoked_when_enabled();

    if (failures > 0) {
        printf("\n=== %d FAILURES ===\n", failures);
        return 1;
    }
    printf("\n=== all tests passed ===\n");
    return 0;
}

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
   main
   ================================================================ */

int main(void) {
    setbuf(stdout, NULL);

    /* Stage 1: env var infrastructure */
    test_compile_disabled_by_default();
    test_compile_enabled_via_env();
    test_compile_explicit_disable();

    if (failures > 0) {
        printf("\n=== %d FAILURES ===\n", failures);
        return 1;
    }
    printf("\n=== all tests passed ===\n");
    return 0;
}

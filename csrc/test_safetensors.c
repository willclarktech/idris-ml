/* Standalone test for SafeTensors serialization round-trip */

#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;

#define ASSERT_NEAR(msg, got, expected, tol) do { \
    double _g = (got), _e = (expected); \
    if (fabs(_g - _e) > (tol)) { \
        printf("FAIL: %s: got %.6f, expected %.6f\n", msg, _g, _e); \
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

int main(void) {
    printf("--- SafeTensors round-trip test ---\n\n");
    param_clear();

    /* Create a 2D param [2, 3] */
    double w_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double* w_buf = tensor_alloc_doubles(6);
    for (int i = 0; i < 6; i++) tensor_write_double(w_buf, i, w_data[i]);
    TensorHandle w = tensor_create_param_2d(2, 3, w_buf);
    param_register("weights", w);

    /* Create a 1D param [2] */
    double b_data[] = {10.0, 20.0};
    double* b_buf = tensor_alloc_doubles(2);
    for (int i = 0; i < 2; i++) tensor_write_double(b_buf, i, b_data[i]);
    TensorHandle b = tensor_create_param_1d(2, b_buf);
    param_register("biases", b);

    ASSERT_TRUE("param_count == 2", param_count() == 2);
    printf("param 0: '%s' (numel=%d, dim=%d)\n", param_name(0), tensor_numel(param_tensor(0)), tensor_dim(param_tensor(0)));
    printf("param 1: '%s' (numel=%d, dim=%d)\n", param_name(1), tensor_numel(param_tensor(1)), tensor_dim(param_tensor(1)));

    /* Verify initial values */
    {
        double* buf = (double*)malloc(6 * sizeof(double));
        tensor_to_doubles(param_tensor(0), buf);
        printf("initial w: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
               buf[0], buf[1], buf[2], buf[3], buf[4], buf[5]);
        ASSERT_NEAR("initial w[0]", buf[0], 1.0, 1e-15);
        ASSERT_NEAR("initial w[5]", buf[5], 6.0, 1e-15);
        free(buf);
    }

    /* Save */
    const char* path = "/tmp/idrisml_test.safetensors";
    int rc = param_save(path);
    ASSERT_TRUE("param_save returns 0", rc == 0);

    /* Check file */
    FILE* f = fopen(path, "rb");
    ASSERT_TRUE("file exists", f != NULL);
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fclose(f);
        printf("file size: %ld bytes (expected: 8 + header + 64 = ~200)\n", sz);
        ASSERT_TRUE("file size > 8", sz > 8);
    }

    /* Corrupt param data to zeros */
    printf("\ncorrupting params to zeros...\n");
    double zeros6[6] = {0};
    param_load_data(0, zeros6, 6);
    double zeros2[2] = {0};
    param_load_data(1, zeros2, 2);
    {
        double* buf = (double*)malloc(6 * sizeof(double));
        tensor_to_doubles(param_tensor(0), buf);
        ASSERT_NEAR("corrupted w[0]", buf[0], 0.0, 1e-15);
        ASSERT_NEAR("corrupted w[5]", buf[5], 0.0, 1e-15);
        free(buf);
    }

    /* Load back */
    printf("\nloading from file...\n");
    rc = param_load(path);
    ASSERT_TRUE("param_load returns 0", rc == 0);

    /* Verify restored values */
    printf("\nverifying restored values...\n");
    {
        double* buf = (double*)malloc(6 * sizeof(double));
        tensor_to_doubles(param_tensor(0), buf);
        for (int i = 0; i < 6; i++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "restored w[%d]", i);
            ASSERT_NEAR(msg, buf[i], w_data[i], 1e-15);
        }
        free(buf);
    }
    {
        double* buf = (double*)malloc(2 * sizeof(double));
        tensor_to_doubles(param_tensor(1), buf);
        ASSERT_NEAR("restored b[0]", buf[0], b_data[0], 1e-15);
        ASSERT_NEAR("restored b[1]", buf[1], b_data[1], 1e-15);
        free(buf);
    }

    /* Clean up */
    remove(path);
    param_clear();

    printf("\n");
    if (failures == 0) {
        printf("All safetensors tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}

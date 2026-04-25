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

    /* --- Optimizer state round-trip --- */
    printf("\n--- Optimizer state round-trip test ---\n\n");

    /* Create an Adam optimizer and run a few steps to populate buffers */
    OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);

    /* Do a few fake training steps to populate optimizer state */
    /* First, we need gradients. Set them manually by doing backward. */
    /* For simplicity, just set up a simple loss and backward */
    {
        TensorHandle loss = tensor_dot(param_tensor(0), param_tensor(0)); /* sum of squares */
        tensor_backward(loss);
        optimizer_step(opt);

        loss = tensor_dot(param_tensor(0), param_tensor(0));
        tensor_backward(loss);
        optimizer_step(opt);
    }

    /* Read back optimizer state */
    double meta[9];
    optimizer_get_meta(opt, meta);
    printf("opt type=%d, lr=%g, t=%d\n", (int)meta[0], meta[1], (int)meta[8]);
    ASSERT_NEAR("opt type", meta[0], 2.0, 1e-15);   /* Adam=2 */
    ASSERT_NEAR("opt lr", meta[1], 0.001, 1e-15);
    ASSERT_NEAR("opt step", meta[8], 2.0, 1e-15);    /* 2 steps */

    double m_buf[6], v_buf[6];
    optimizer_get_m(opt, 0, m_buf);
    optimizer_get_v(opt, 0, v_buf);
    printf("m[0]=%.10f, v[0]=%.10f\n", m_buf[0], v_buf[0]);
    ASSERT_TRUE("m[0] != 0 (populated)", m_buf[0] != 0.0);
    ASSERT_TRUE("v[0] != 0 (populated)", v_buf[0] != 0.0);

    /* Save optimizer state */
    const char* opt_path = "/tmp/idrisml_test.optimizer.safetensors";
    int orc = optimizer_save(opt, opt_path);
    ASSERT_TRUE("optimizer_save returns 0", orc == 0);

    /* Create a fresh optimizer and load state */
    OptimizerHandle opt2 = optimizer_create_adam(0.1, 0.5, 0.5, 0.1); /* different params */
    orc = optimizer_load(opt2, opt_path);
    ASSERT_TRUE("optimizer_load returns 0", orc == 0);

    /* Verify restored meta */
    double meta2[9];
    optimizer_get_meta(opt2, meta2);
    ASSERT_NEAR("restored opt type", meta2[0], 2.0, 1e-15);
    ASSERT_NEAR("restored opt lr", meta2[1], 0.001, 1e-15);
    ASSERT_NEAR("restored opt step", meta2[8], 2.0, 1e-15);

    /* Verify restored buffers */
    double m_buf2[6], v_buf2[6];
    optimizer_get_m(opt2, 0, m_buf2);
    optimizer_get_v(opt2, 0, v_buf2);
    for (int i = 0; i < 6; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "restored opt m[%d]", i);
        ASSERT_NEAR(msg, m_buf2[i], m_buf[i], 1e-15);
        snprintf(msg, sizeof(msg), "restored opt v[%d]", i);
        ASSERT_NEAR(msg, v_buf2[i], v_buf[i], 1e-15);
    }

    /* Clean up */
    remove(opt_path);
    optimizer_free(opt);
    optimizer_free(opt2);
    param_clear();

    printf("\n");
    if (failures == 0) {
        printf("All safetensors tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}

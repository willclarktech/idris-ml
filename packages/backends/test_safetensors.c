/* Standalone test for SafeTensors serialization round-trip */

#include "backend.h"
#include "shared_utils.h"
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

#ifdef BACKEND_TORCH
/* Inference-dtype create via the unified dtag-dispatch symbol (dtags from
   the Idris RuntimeDType: F32=0, F64=1, BF16=2, F16=3, I8=4, I16=5, I32=6,
   I64=7, U8=8, Bool=9). The per-dtype create symbols were retired in the
   Phase 1 unification (commit 4b77f9a). */

/* Save -> zero -> load round-trip for one inference dtype. `data` holds
   values that are exactly representable in the dtype, so the restored
   values must match bit-for-bit (the create call already quantized). */
static void dtype_roundtrip(const char* label, const char* expect_dtype,
                            TensorHandle h, double* data, int n) {
    char msg[80];
    param_clear();
    param_register(label, h);

    snprintf(msg, sizeof(msg), "%s: on-disk dtype is %s", label, expect_dtype);
    ASSERT_TRUE(msg, strcmp(tensor_dtype_name(param_tensor(0)), expect_dtype) == 0);

    const char* path = "/tmp/idrisml_test_dtype.safetensors";
    snprintf(msg, sizeof(msg), "%s: param_save returns 0", label);
    ASSERT_TRUE(msg, param_save(path) == 0);

    double* zeros = (double*)calloc(n, sizeof(double));
    param_load_data(0, zeros, n);
    free(zeros);

    snprintf(msg, sizeof(msg), "%s: param_load returns 0", label);
    ASSERT_TRUE(msg, param_load(path) == 0);

    double* got = (double*)malloc(n * sizeof(double));
    tensor_to_doubles(param_tensor(0), got);
    for (int i = 0; i < n; i++) {
        snprintf(msg, sizeof(msg), "%s: restored [%d]", label, i);
        ASSERT_NEAR(msg, got[i], data[i], 1e-9);
    }
    free(got);
    remove(path);
    param_clear();
}

static void run_inference_dtype_tests(void) {
    printf("\n--- Inference-dtype safetensors round-trip (torch) ---\n\n");
    /* Exactly representable in bf16/f16/i32: small ints + simple binary
       fractions (1.5 = 1.1b, 256 = 2^8, -0.5 = -2^-1). */
    double fdata[] = {1.0, -2.0, 1.5, 256.0, -0.5, 0.0};
    int    fn = 6;
    double idata[] = {1.0, -2.0, 3.0, 1000.0, -42.0, 0.0};
    int    in = 6;

    double* b1 = tensor_alloc_doubles(fn);
    for (int i = 0; i < fn; i++) b1[i] = fdata[i];
    dtype_roundtrip("w_bf16", "BF16", tensor_create_1d_streamed(fn, b1, 0, 0, 2), fdata, fn);

    double* b2 = tensor_alloc_doubles(fn);
    for (int i = 0; i < fn; i++) b2[i] = fdata[i];
    dtype_roundtrip("w_f16", "F16", tensor_create_1d_streamed(fn, b2, 0, 0, 3), fdata, fn);

    double* b3 = tensor_alloc_doubles(in);
    for (int i = 0; i < in; i++) b3[i] = idata[i];
    dtype_roundtrip("w_i32", "I32", tensor_create_1d_streamed(in, b3, 0, 0, 6), idata, in);
}
#endif /* BACKEND_TORCH */

int main(void) {
    printf("--- SafeTensors round-trip test ---\n\n");
    param_clear();

    /* Create a 2D param [2, 3] */
    double w_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double* w_buf = tensor_alloc_doubles(6);
    for (int i = 0; i < 6; i++) w_buf[i] = w_data[i];
    TensorHandle w = tensor_create_param_2d(2, 3, w_buf);
    param_register("weights", w);

    /* Create a 1D param [2] */
    double b_data[] = {10.0, 20.0};
    double* b_buf = tensor_alloc_doubles(2);
    for (int i = 0; i < 2; i++) b_buf[i] = b_data[i];
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
        /* sum(w*w) — sum-of-squares, portable across backends (torch's
           tensor_dot requires 1-D, and param 0 is 2-D [2,3]). */
        TensorHandle loss = tensor_sum(tensor_mul(param_tensor(0), param_tensor(0)));
        tensor_backward(loss);
        optimizer_step(opt);

        loss = tensor_sum(tensor_mul(param_tensor(0), param_tensor(0)));
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

#ifdef BACKEND_TORCH
    run_inference_dtype_tests();
#endif

    printf("\n");
    if (failures == 0) {
        printf("All safetensors tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}

/* Dtype-scaffolding Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"


Test(dtype_scaffolding, unified_dtag_dispatch) {
    param_clear();

    double* d = (double*)malloc(3 * sizeof(double));
    d[0] = 1.5; d[1] = 2.5; d[2] = 3.5;
    TensorHandle t = tensor_create_1d_streamed(3, d, 0, 0, 15);  /* dtag 15 = F64 */
    ASSERT_TRUE("unified create_1d dtag=1 -> F64",
                strcmp(tensor_dtype_name(t), "F64") == 0);
    double out[3];
    tensor_to_doubles(t, out);
    ASSERT_NEAR("unified create_1d[0]", out[0], 1.5, 1e-10);
    ASSERT_NEAR("unified create_1d[2]", out[2], 3.5, 1e-10);

    /* cast via the unified symbol, dtag=1 — identity on an F64 source. */
    TensorHandle c = tensor_cast_dtype_streamed(t, 0, 15);
    ASSERT_TRUE("unified cast dtag=1 -> F64",
                strcmp(tensor_dtype_name(c), "F64") == 0);
    double cout[3];
    tensor_to_doubles(c, cout);
    ASSERT_NEAR("unified cast preserves[1]", cout[1], 2.5, 1e-10);

    /* scalar via the unified symbol, dtag=1. */
    TensorHandle s = tensor_create_scalar_streamed(7.0, 0, 0, 15);
    ASSERT_NEAR("unified scalar dtag=1", tensor_item(s), 7.0, 1e-10);

#if defined(BACKEND_TORCH)
    /* torch reaches the inference dtags through the same unified symbol. */
    double* di = (double*)malloc(2 * sizeof(double));
    di[0] = 5.0; di[1] = 6.0;
    TensorHandle ti = tensor_create_1d_streamed(2, di, 0, 0, 10);  /* dtag 10 = I32 */
    ASSERT_TRUE("unified create_1d dtag=6 -> I32",
                strcmp(tensor_dtype_name(ti), "I32") == 0);
#endif
    param_clear();
}

Test(dtype_scaffolding, runtime_dtype_tag_layout) {
    param_clear();

    struct { int dtag; const char* name; } universal_cases[] = {
        { 14, "F32" },
        { 15, "F64" },
    };
    for (size_t i = 0; i < sizeof(universal_cases)/sizeof(universal_cases[0]); i++) {
        TensorHandle h = tensor_create_scalar_streamed(1.0, 0, 0, universal_cases[i].dtag);
        const char* got = tensor_dtype_name(h);
        char label[128];
        snprintf(label, sizeof(label), "dtag=%d expected %s got %s",
                 universal_cases[i].dtag, universal_cases[i].name, got);
        ASSERT_TRUE(label, strcmp(got, universal_cases[i].name) == 0);
    }

#if defined(BACKEND_TORCH) || defined(BACKEND_TAPE)
    /* Inference dtags — torch wires every dtype; tape stores them via
       the double lingua franca (Phase 2). mlx supports only F32/F64
       on Metal, so its inference cases are gated above. */
    struct { int dtag; const char* name; } inference_cases[] = {
        { 1,  "Bool" },
        { 4,  "U8" },
        { 8,  "I8" },
        { 9,  "I16" },
        { 10, "I32" },
        { 11, "I64" },
        { 13, "F16" },
        { 17, "BF16" },
    };
    for (size_t i = 0; i < sizeof(inference_cases)/sizeof(inference_cases[0]); i++) {
        TensorHandle h = tensor_create_scalar_streamed(1.0, 0, 0, inference_cases[i].dtag);
        const char* got = tensor_dtype_name(h);
        char label[128];
        const char* expected = inference_cases[i].name;
        /* tape stringifies Bool as "BOOL"; torch as "Bool". Accept both. */
        int match = (strcmp(got, expected) == 0);
        if (!match && strcmp(expected, "Bool") == 0 && strcmp(got, "BOOL") == 0) match = 1;
        snprintf(label, sizeof(label), "dtag=%d expected %s got %s",
                 inference_cases[i].dtag, expected, got);
        ASSERT_TRUE(label, match);
    }
#endif
    param_clear();
}

/* Test NTM addressing chain backward at realistic scale.
   Isolates the NaN gradient issue from the Idris FFI layer. */

#include "backend.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 128  /* memory slots */
#define W 20   /* memory width */

int main(void) {
    param_clear();
    srand(42);

    /* Create memory matrix [N, W] as a param */
    double* mem_data = calloc(N * W, sizeof(double));
    for (int i = 0; i < N * W; i++) mem_data[i] = 1e-6; /* small init */
    TensorHandle mem = tensor_create_param_2d(N, W, mem_data);
    param_register("mem", mem);
    free(mem_data);

    /* Key vector [W] as a param */
    double* key_data = calloc(W, sizeof(double));
    for (int i = 0; i < W; i++) key_data[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    TensorHandle key = tensor_create_param_1d(W, key_data);
    param_register("key", key);
    free(key_data);

    /* Previous addressing weights [N] — uniform */
    double* prev_w_data = calloc(N, sizeof(double));
    for (int i = 0; i < N; i++) prev_w_data[i] = 1.0 / N;
    TensorHandle prev_w = tensor_create_param_1d(N, prev_w_data);
    param_register("prev_w", prev_w);
    free(prev_w_data);

    /* Scalars: beta, g, gamma */
    TensorHandle beta = tensor_create_scalar(1.0, 1);
    TensorHandle g = tensor_create_scalar(0.5, 1);
    TensorHandle gamma = tensor_create_scalar(1.5, 1);
    param_register("beta", beta);
    param_register("g", g);
    param_register("gamma", gamma);

    /* Shift kernel [3] — center-heavy */
    double sk_data[] = {0.1, 0.8, 0.1};
    TensorHandle shift_k = tensor_create_param_1d(3, sk_data);
    param_register("shift", shift_k);

    printf("Params registered: %d\n", param_count());

    /* === NTM addressing pipeline === */

    /* 1. Cosine similarity: [N, W] vs [1, W] → [N] */
    TensorHandle key_exp = tensor_unsqueeze(key, 0);  /* [1, W] */
    TensorHandle cos_sim = tensor_cosine_similarity(mem, key_exp, 1);
    printf("cos_sim[0] = %f, rg=%d\n", tensor_item(tensor_select(cos_sim, 0, 0)), tensor_requires_grad(cos_sim));

    /* 2. Scale by beta */
    TensorHandle scaled = tensor_mul(beta, cos_sim);  /* broadcast: scalar * [N] */
    printf("scaled[0] = %f\n", tensor_item(tensor_select(scaled, 0, 0)));

    /* 3. Softmax → content weights */
    TensorHandle content = tensor_softmax(scaled, 0);
    printf("content[0] = %f\n", tensor_item(tensor_select(content, 0, 0)));

    /* 4. Interpolation: g * content + (1-g) * prev_w */
    TensorHandle g_content = tensor_mul(g, content);
    TensorHandle one_minus_g = tensor_sub(tensor_create_scalar(1.0, 0), g);
    TensorHandle omg_prev = tensor_mul(one_minus_g, prev_w);
    TensorHandle interp = tensor_add(g_content, omg_prev);
    printf("interp[0] = %f\n", tensor_item(tensor_select(interp, 0, 0)));

    /* 5. Shift (circular conv) */
    TensorHandle shifted = tensor_conv1d_circular(interp, shift_k);
    printf("shifted[0] = %f\n", tensor_item(tensor_select(shifted, 0, 0)));

    /* 6. Clamp + Focus (sharpen) */
    TensorHandle clamped = tensor_clamp_min(shifted, 1e-10);
    TensorHandle powered = tensor_pow(clamped, gamma);
    TensorHandle pow_sum = tensor_sum(powered);
    TensorHandle pow_sum_eps = tensor_add_scalar(pow_sum, 1e-10);
    TensorHandle focused = tensor_div(powered, pow_sum_eps);
    printf("focused[0] = %f, sum=%f\n",
           tensor_item(tensor_select(focused, 0, 0)),
           tensor_item(tensor_sum(focused)));

    /* 7. Read: focused @ mem → [W] */
    TensorHandle read_out = tensor_matmul(focused, mem);
    printf("read_out[0] = %f\n", tensor_item(tensor_select(read_out, 0, 0)));

    /* 8. Loss = sum(read_out) */
    TensorHandle loss = tensor_sum(read_out);
    printf("loss = %f, rg=%d\n", tensor_item(loss), tensor_requires_grad(loss));

    /* === Backward === */
    tensor_backward(loss);

    /* Check for NaN gradients */
    int nan_count = 0, total = 0;
    for (int i = 0; i < param_count(); i++) {
        double g_val = param_grad_item(i);
        total++;
        if (g_val != g_val) nan_count++;
    }
    printf("\nNaN gradients: %d / %d\n", nan_count, total);

    /* Print key gradients */
    printf("grad beta = %f\n", param_grad_item(3));
    printf("grad g = %f\n", param_grad_item(4));
    printf("grad gamma = %f\n", param_grad_item(5));
    printf("grad mem[0] = %f\n", param_grad_item(0));
    printf("grad key[0] = %f\n", param_grad_item(1));

    if (nan_count == 0) {
        printf("\nAll NTM addressing gradients are finite!\n");
    } else {
        printf("\nFAILED: %d NaN gradients\n", nan_count);
    }

    return nan_count > 0 ? 1 : 0;
}

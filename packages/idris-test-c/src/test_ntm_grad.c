/* Test NTM addressing chain backward at realistic scale.
 * Isolates the NaN gradient issue from the Idris FFI layer.
 *
 * Single integration Test() — the file is 100+ lines of sequential
 * addressing-chain construction (cosine similarity → softmax →
 * interpolation → shift → clamp → focus → read → loss → backward),
 * and decomposing into multiple Test() cases would force replicating
 * the param setup chain. Criterion forks per Test(), so the
 * cross-test global registry isn't an issue — each fork starts fresh.
 */
#include <criterion/criterion.h>
#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 128 /* memory slots */
#define W 20  /* memory width */

Test(ntm_grad, addressing_chain_no_nans) {
	param_clear();
	srand(42);

	/* Create memory matrix [N, W] as a param */
	/* Note: tensor_create_param_* takes ownership and frees the buffer */
	double* mem_data = (double*)malloc(N * W * sizeof(double));
	for (int i = 0; i < N * W; i++)
		mem_data[i] = 1e-6; /* small init */
	TensorHandle mem = tensor_create_param_2d_f64(N, W, mem_data);
	param_register("mem", mem);

	/* Key vector [W] as a param */
	double* key_data = (double*)malloc(W * sizeof(double));
	for (int i = 0; i < W; i++)
		key_data[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
	TensorHandle key = tensor_create_param_1d_f64(W, key_data);
	param_register("key", key);

	/* Previous addressing weights [N] — uniform */
	double* prev_w_data = (double*)malloc(N * sizeof(double));
	for (int i = 0; i < N; i++)
		prev_w_data[i] = 1.0 / N;
	TensorHandle prev_w = tensor_create_param_1d_f64(N, prev_w_data);
	param_register("prev_w", prev_w);

	/* Scalars: beta, g, gamma */
	TensorHandle beta = tensor_create_scalar(1.0, 1);
	TensorHandle g = tensor_create_scalar(0.5, 1);
	TensorHandle gamma = tensor_create_scalar(1.5, 1);
	param_register("beta", beta);
	param_register("g", g);
	param_register("gamma", gamma);

	/* Shift kernel [3] — center-heavy */
	double sk_src[] = {0.1, 0.8, 0.1};
	double* sk_data = (double*)malloc(3 * sizeof(double));
	memcpy(sk_data, sk_src, 3 * sizeof(double));
	TensorHandle shift_k = tensor_create_param_1d_f64(3, sk_data);
	param_register("shift", shift_k);

	/* === NTM addressing pipeline === */

	/* 1. Cosine similarity: [N, W] vs [1, W] → [N] */
	TensorHandle key_exp = tensor_unsqueeze(key, 0); /* [1, W] */
	TensorHandle cos_sim = tensor_cosine_similarity(mem, key_exp, 1);

	/* 2. Scale by beta */
	TensorHandle scaled = tensor_mul(beta, cos_sim); /* broadcast: scalar * [N] */

	/* 3. Softmax → content weights */
	TensorHandle content = tensor_softmax(scaled, 0);

	/* 4. Interpolation: g * content + (1-g) * prev_w */
	TensorHandle g_content = tensor_mul(g, content);
	TensorHandle one_minus_g = tensor_sub(tensor_create_scalar(1.0, 0), g);
	TensorHandle omg_prev = tensor_mul(one_minus_g, prev_w);
	TensorHandle interp = tensor_add(g_content, omg_prev);

	/* 5. Shift (circular conv) */
	TensorHandle shifted = tensor_conv1d_circular(interp, shift_k);

	/* 6. Clamp + Focus (sharpen) */
	TensorHandle clamped = tensor_clamp_min(shifted, 1e-10);
	TensorHandle powered = tensor_pow(clamped, gamma);
	TensorHandle pow_sum = tensor_sum(powered);
	TensorHandle pow_sum_eps = tensor_add_scalar(pow_sum, 1e-10);
	TensorHandle focused = tensor_div(powered, pow_sum_eps);

	/* 7. Read: focused @ mem → [W] */
	TensorHandle read_out = tensor_matmul(focused, mem);

	/* 8. Loss = sum(read_out) */
	TensorHandle loss = tensor_sum(read_out);

	/* === Backward === */
	tensor_backward(loss);

	/* Check for NaN gradients */
	int nan_count = 0, total = 0;
	for (int i = 0; i < param_count(); i++) {
		double g_val = param_grad_item(i);
		total++;
		if (g_val != g_val) nan_count++;
	}

	cr_assert_eq(nan_count, 0, "NTM addressing chain produced %d NaN gradients out of %d params",
	             nan_count, total);
}

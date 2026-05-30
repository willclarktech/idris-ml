/* Test a single NTM timestep: LSTM → FC → addressing → output.
 * Mirrors the Idris NTM applyVar scalar fallback path.
 * Goal: find where NaN enters when combining LSTM + addressing.
 *
 * Single integration Test() — the file is 200+ lines of sequential
 * LSTM+FC+addressing+backward setup. Decomposing into multiple Test()
 * cases would force replicating the 130-line scalar-param allocation
 * chain (LSTM iw/rw/bias + h0/c0 + read FC weights + memory + addressing
 * weights). Criterion forks per Test(), so the cross-test global
 * registry isn't an issue — each fork starts fresh.
 */
#include <criterion/criterion.h>
#include "backend.h"
#include "shared_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 128    /* memory slots */
#define M 20     /* memory width */
#define H 100    /* controller hidden size */
#define INP 9    /* input width (W=8 + delimiter) */
#define GATE (4*H)  /* LSTM gate size */

/* Helper: create a vector of scalar params */
static void create_scalar_params(const char* prefix, int count, double init,
                                  TensorHandle* out) {
    for (int i = 0; i < count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s_%d", prefix, i);
        out[i] = tensor_create_scalar(init + ((double)(i % 7) - 3) * 0.01, 1);
        param_register(name, out[i]);
    }
}

/* Helper: stack scalar handles into a 1D tensor */
static TensorHandle stack_scalars(TensorHandle* handles, int n) {
    void** arr = tensor_ptr_array_alloc(n);
    for (int i = 0; i < n; i++) tensor_ptr_array_set_return(arr, i, handles[i]);
    return tensor_stack_from_array(arr, n, 0);
}

/* Helper: reshape [n] to [rows, cols] */
static TensorHandle reshape_2d(TensorHandle t, int rows, int cols) {
    int shape[] = {rows, cols};
    return tensor_reshape(t, shape, 2);
}

/* Helper: check for NaN in all param grads */
static int count_nan_grads(void) {
    int nan_count = 0;
    for (int i = 0; i < param_count(); i++) {
        double g = param_grad_item(i);
        if (g != g) nan_count++;
    }
    return nan_count;
}

Test(ntm_timestep, lstm_addressing_backward_no_nans) {
    param_clear();
    srand(42);

    /* --- Create all parameters --- */

    /* LSTM weights: inputWeights [GATE, M+INP], recurrentWeights [GATE, H], bias [GATE] */
    int lstm_iw_size = GATE * (M + INP);
    int lstm_rw_size = GATE * H;
    TensorHandle* lstm_iw = calloc(lstm_iw_size, sizeof(TensorHandle));
    TensorHandle* lstm_rw = calloc(lstm_rw_size, sizeof(TensorHandle));
    TensorHandle* lstm_b  = calloc(GATE, sizeof(TensorHandle));
    TensorHandle* lstm_h0 = calloc(H, sizeof(TensorHandle));
    TensorHandle* lstm_c0 = calloc(H, sizeof(TensorHandle));
    create_scalar_params("iw", lstm_iw_size, 0.0, lstm_iw);
    create_scalar_params("rw", lstm_rw_size, 0.0, lstm_rw);
    create_scalar_params("bias", GATE, 0.0, lstm_b);
    create_scalar_params("h0", H, 0.01, lstm_h0);
    create_scalar_params("c0", H, 0.01, lstm_c0);

    /* Read FC: [ReadParamWidth, H] = [(M+3+3), H] */
    int read_param_w = M + 3 + 3;  /* key(M) + shift(3) + beta + g + gamma */
    int rfc_size = read_param_w * H;
    TensorHandle* rfc_w = calloc(rfc_size, sizeof(TensorHandle));
    TensorHandle* rfc_b = calloc(read_param_w, sizeof(TensorHandle));
    create_scalar_params("rfc_w", rfc_size, 0.0, rfc_w);
    create_scalar_params("rfc_b", read_param_w, 0.0, rfc_b);

    /* Memory [N, M] */
    TensorHandle* mem = calloc(N * M, sizeof(TensorHandle));
    create_scalar_params("mem", N * M, 1e-6, mem);

    /* Addressing weights [N] */
    TensorHandle* read_addr = calloc(N, sizeof(TensorHandle));
    create_scalar_params("raddr", N, 1.0 / N, read_addr);

    /* Read output [M] */
    TensorHandle* read_out = calloc(M, sizeof(TensorHandle));
    create_scalar_params("rout", M, 0.0, read_out);

    int total_params = param_count();

    /* --- Input (not a param) --- */
    TensorHandle* input = calloc(INP, sizeof(TensorHandle));
    for (int i = 0; i < INP; i++) input[i] = tensor_create_scalar(i < 4 ? 1.0 : 0.0, 0);

    /* === LSTM Forward === */

    /* Concatenate readOutput ++ input → [M+INP] */
    int lstm_inp_size = M + INP;
    TensorHandle* lstm_inp = calloc(lstm_inp_size, sizeof(TensorHandle));
    for (int i = 0; i < M; i++) lstm_inp[i] = read_out[i];
    for (int i = 0; i < INP; i++) lstm_inp[M + i] = input[i];

    /* Stack into tensors */
    TensorHandle iw_t = reshape_2d(stack_scalars(lstm_iw, lstm_iw_size), GATE, lstm_inp_size);
    TensorHandle rw_t = reshape_2d(stack_scalars(lstm_rw, lstm_rw_size), GATE, H);
    TensorHandle b_t = stack_scalars(lstm_b, GATE);
    TensorHandle h_t = stack_scalars(lstm_h0, H);
    TensorHandle c_t = stack_scalars(lstm_c0, H);
    TensorHandle x_t = stack_scalars(lstm_inp, lstm_inp_size);

    /* LSTM gate computation: combined = iw @ x + rw @ h + b */
    TensorHandle mulIW = tensor_mv(iw_t, x_t);
    TensorHandle mulRW = tensor_mv(rw_t, h_t);
    TensorHandle combined = tensor_add(tensor_add(mulIW, mulRW), b_t);

    /* LSTM gates via pair */
    TensorPair* gates = tensor_lstm_gates_pair(combined, c_t, H);
    TensorHandle new_c = tensor_pair_second(gates);

    /* === Read FC: cell → readParams === */
    TensorHandle rfc_wt = reshape_2d(stack_scalars(rfc_w, rfc_size), read_param_w, H);
    TensorHandle rfc_bt = stack_scalars(rfc_b, read_param_w);
    TensorHandle read_params = tensor_add(tensor_mv(rfc_wt, new_c), rfc_bt);

    /* Parse read params: key[M], shift[3], beta, g, gamma */
    /* Select elements from read_params to form addressing inputs */
    TensorHandle* key_elems = calloc(M, sizeof(TensorHandle));
    for (int i = 0; i < M; i++) key_elems[i] = tensor_select(read_params, 0, i);
    TensorHandle key_vec = stack_scalars(key_elems, M);

    TensorHandle* shift_elems = calloc(3, sizeof(TensorHandle));
    for (int i = 0; i < 3; i++) shift_elems[i] = tensor_select(read_params, 0, M + i);
    TensorHandle shift_vec = tensor_softmax(stack_scalars(shift_elems, 3), 0);

    TensorHandle beta_v = tensor_select(read_params, 0, M + 3);
    /* softplus: log(1 + exp(x)) */
    TensorHandle beta_sp = tensor_log(tensor_add_scalar(tensor_exp(beta_v), 1.0));

    TensorHandle g_v = tensor_sigmoid(tensor_select(read_params, 0, M + 4));

    TensorHandle gamma_raw = tensor_select(read_params, 0, M + 5);
    TensorHandle gamma_v = tensor_add_scalar(tensor_log(tensor_add_scalar(tensor_exp(gamma_raw), 1.0)), 1.0);

    /* Run addressing chain */
    TensorHandle mem_t = reshape_2d(stack_scalars(mem, N * M), N, M);
    TensorHandle raddr_t = stack_scalars(read_addr, N);

    TensorHandle key_exp = tensor_unsqueeze(key_vec, 0);
    TensorHandle cos_sim = tensor_cosine_similarity(mem_t, key_exp, 1);
    TensorHandle scaled = tensor_mul(beta_sp, cos_sim);
    TensorHandle content = tensor_softmax(scaled, 0);
    TensorHandle g_content = tensor_mul(g_v, content);
    TensorHandle omg = tensor_sub(tensor_create_scalar(1.0, 0), g_v);
    TensorHandle omg_prev = tensor_mul(omg, raddr_t);
    TensorHandle interp = tensor_add(g_content, omg_prev);
    TensorHandle shifted = tensor_conv1d_circular(interp, shift_vec);
    TensorHandle clamped = tensor_clamp_min(shifted, 1e-10);
    TensorHandle powered = tensor_pow(clamped, gamma_v);
    TensorHandle pow_sum = tensor_add_scalar(tensor_sum(powered), 1e-10);
    TensorHandle focused = tensor_div(powered, pow_sum);
    TensorHandle read_result = tensor_matmul(focused, mem_t);

    TensorHandle loss = tensor_sum(read_result);

    free(key_elems); free(shift_elems);

    /* === Backward === */
    tensor_backward(loss);

    int nan_grads = count_nan_grads();

    /* Cleanup */
    free(lstm_iw); free(lstm_rw); free(lstm_b); free(lstm_h0); free(lstm_c0);
    free(rfc_w); free(rfc_b); free(mem); free(read_addr); free(read_out);
    free(input); free(lstm_inp);

    cr_assert_eq(nan_grads, 0,
        "NTM timestep (LSTM + FC + addressing) produced %d NaN gradients out of %d params",
        nan_grads, total_params);
}

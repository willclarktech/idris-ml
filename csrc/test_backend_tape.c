/* Test suite for the tape-based C backend (backend_tape.c).
   Tests the same backend.h API as test_backend.c but links against
   the tape backend instead of libtorch. */

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

/* ================================================================
   T1: Scalar tensor creation + arithmetic
   ================================================================ */

static void test_scalar_creation(void) {
    printf("\n--- Scalar creation ---\n");
    TensorHandle a = tensor_create_scalar(3.0, 0);
    ASSERT_NEAR("scalar item", tensor_item(a), 3.0, 1e-10);
    ASSERT_NEAR("numel", (double)tensor_numel(a), 1.0, 1e-10);
    ASSERT_NEAR("dim", (double)tensor_dim(a), 0.0, 1e-10);
    ASSERT_TRUE("requires_grad=0", tensor_requires_grad(a) == 0);

    TensorHandle b = tensor_create_scalar(4.0, 1);
    ASSERT_TRUE("requires_grad=1", tensor_requires_grad(b) == 1);

    tensor_free(a);
    tensor_free(b);
}

static void test_arithmetic(void) {
    printf("\n--- Arithmetic ---\n");
    TensorHandle a = tensor_create_scalar(3.0, 0);
    TensorHandle b = tensor_create_scalar(4.0, 0);

    TensorHandle c;
    c = tensor_add(a, b);
    ASSERT_NEAR("3+4", tensor_item(c), 7.0, 1e-10);
    tensor_free(c);

    c = tensor_sub(a, b);
    ASSERT_NEAR("3-4", tensor_item(c), -1.0, 1e-10);
    tensor_free(c);

    c = tensor_mul(a, b);
    ASSERT_NEAR("3*4", tensor_item(c), 12.0, 1e-10);
    tensor_free(c);

    c = tensor_div(a, b);
    ASSERT_NEAR("3/4", tensor_item(c), 0.75, 1e-10);
    tensor_free(c);

    c = tensor_neg(a);
    ASSERT_NEAR("neg(3)", tensor_item(c), -3.0, 1e-10);
    tensor_free(c);

    c = tensor_abs(tensor_create_scalar(-5.0, 0));
    ASSERT_NEAR("abs(-5)", tensor_item(c), 5.0, 1e-10);
    tensor_free(c);

    c = tensor_exp(a);
    ASSERT_NEAR("exp(3)", tensor_item(c), exp(3.0), 1e-6);
    tensor_free(c);

    c = tensor_log(b);
    ASSERT_NEAR("log(4)", tensor_item(c), log(4.0), 1e-10);
    tensor_free(c);

    c = tensor_sqrt(b);
    ASSERT_NEAR("sqrt(4)", tensor_item(c), 2.0, 1e-10);
    tensor_free(c);

    c = tensor_pow(a, tensor_create_scalar(2.0, 0));
    ASSERT_NEAR("3^2", tensor_item(c), 9.0, 1e-10);
    tensor_free(c);

    c = tensor_sigmoid(a);
    ASSERT_NEAR("sigmoid(3)", tensor_item(c), 1.0/(1.0+exp(-3.0)), 1e-10);
    tensor_free(c);

    c = tensor_tanh(a);
    ASSERT_NEAR("tanh(3)", tensor_item(c), tanh(3.0), 1e-10);
    tensor_free(c);

    c = tensor_add_scalar(a, 10.0);
    ASSERT_NEAR("3+10", tensor_item(c), 13.0, 1e-10);
    tensor_free(c);

    c = tensor_mul_scalar(a, 2.5);
    ASSERT_NEAR("3*2.5", tensor_item(c), 7.5, 1e-10);
    tensor_free(c);

    tensor_free(a);
    tensor_free(b);
}

/* ================================================================
   T2: Autograd — backward pass + gradient collection
   ================================================================ */

static void test_autograd_basic(void) {
    printf("\n--- Autograd: y = w*x + b ---\n");
    param_clear();

    TensorHandle w = tensor_create_scalar(3.0, 1);
    TensorHandle b = tensor_create_scalar(1.0, 1);
    param_register("w", w);
    param_register("b", b);

    TensorHandle x = tensor_create_scalar(2.0, 0);
    TensorHandle wx = tensor_mul(w, x);
    TensorHandle y = tensor_add(wx, b);

    tensor_backward(y);

    /* dy/dw = x = 2, dy/db = 1 */
    ASSERT_NEAR("grad w = x", param_grad_item(0), 2.0, 1e-10);
    ASSERT_NEAR("grad b = 1", param_grad_item(1), 1.0, 1e-10);

    /* grad_item_and_zero reads then zeros */
    double gw = param_grad_item_and_zero(0);
    ASSERT_NEAR("grad_and_zero w", gw, 2.0, 1e-10);
    ASSERT_NEAR("zeroed w", param_grad_item(0), 0.0, 1e-10);

    tensor_free(w); tensor_free(b); tensor_free(x);
    tensor_free(wx); tensor_free(y);
    param_clear();
}

static void test_autograd_chain(void) {
    printf("\n--- Autograd: f = (a+b)^2 ---\n");
    param_clear();

    TensorHandle a = tensor_create_scalar(1.0, 1);
    TensorHandle b = tensor_create_scalar(2.0, 1);
    param_register("a", a);
    param_register("b", b);

    TensorHandle s = tensor_add(a, b);    /* s = 3 */
    TensorHandle s2 = tensor_mul(s, s);   /* s2 = 9 */
    tensor_backward(s2);

    /* d/da (a+b)^2 = 2(a+b) = 6 */
    ASSERT_NEAR("d((a+b)^2)/da = 6", param_grad_item_and_zero(0), 6.0, 1e-10);
    ASSERT_NEAR("d((a+b)^2)/db = 6", param_grad_item_and_zero(1), 6.0, 1e-10);

    tensor_free(a); tensor_free(b); tensor_free(s); tensor_free(s2);
    param_clear();
}

static void test_autograd_exp(void) {
    printf("\n--- Autograd: y = exp(w) ---\n");
    param_clear();

    TensorHandle w = tensor_create_scalar(1.0, 1);
    param_register("w", w);
    TensorHandle y = tensor_exp(w);
    tensor_backward(y);

    ASSERT_NEAR("d(exp(w))/dw = exp(1)", param_grad_item_and_zero(0), exp(1.0), 1e-10);

    tensor_free(w); tensor_free(y);
    param_clear();
}

static void test_autograd_div(void) {
    printf("\n--- Autograd: y = a/b ---\n");
    param_clear();

    TensorHandle a = tensor_create_scalar(6.0, 1);
    TensorHandle b = tensor_create_scalar(3.0, 1);
    param_register("a", a);
    param_register("b", b);

    TensorHandle y = tensor_div(a, b);
    tensor_backward(y);

    /* dy/da = 1/b = 1/3, dy/db = -a/b^2 = -6/9 */
    ASSERT_NEAR("da = 1/b", param_grad_item_and_zero(0), 1.0/3.0, 1e-10);
    ASSERT_NEAR("db = -a/b^2", param_grad_item_and_zero(1), -6.0/9.0, 1e-10);

    tensor_free(a); tensor_free(b); tensor_free(y);
    param_clear();
}

static void test_autograd_sqrt(void) {
    printf("\n--- Autograd: y = sqrt(w) ---\n");
    param_clear();

    TensorHandle w = tensor_create_scalar(4.0, 1);
    param_register("w", w);
    TensorHandle y = tensor_sqrt(w);
    tensor_backward(y);

    ASSERT_NEAR("d(sqrt(w))/dw = 1/(2*sqrt(w))", param_grad_item_and_zero(0), 0.25, 1e-10);

    tensor_free(w); tensor_free(y);
    param_clear();
}

static void test_autograd_native_sgd(void) {
    printf("\n--- Native SGD optimizer ---\n");
    param_clear();

    TensorHandle w = tensor_create_scalar(0.5, 1);
    param_register("w", w);

    OptimizerHandle sgd = optimizer_create_sgd(0.01);

    /* Train 100 steps: loss = (w*2 - 3)^2, optimal w = 1.5 */
    for (int step = 0; step < 100; step++) {
        optimizer_zero_grad(sgd);
        TensorHandle x = tensor_create_scalar(2.0, 0);
        TensorHandle t = tensor_create_scalar(3.0, 0);
        TensorHandle pred = tensor_mul(w, x);
        TensorHandle diff = tensor_sub(pred, t);
        TensorHandle loss = tensor_mul(diff, diff);
        tensor_backward(loss);
        optimizer_step(sgd);
        tensor_free(x); tensor_free(t); tensor_free(pred);
        tensor_free(diff); tensor_free(loss);
    }
    ASSERT_NEAR("SGD converges w->1.5", tensor_item(w), 1.5, 0.01);

    optimizer_free(sgd);
    tensor_free(w);
    param_clear();
}

/* ================================================================
   T3: Multi-dimensional tensors + linalg
   ================================================================ */

static void test_multidim(void) {
    printf("\n--- Multi-dim tensors ---\n");

    double mat_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int mat_shape[] = {2, 3};
    TensorHandle mat = tensor_create(mat_data, mat_shape, 2, 0);
    ASSERT_NEAR("mat dim", (double)tensor_dim(mat), 2.0, 1e-10);
    ASSERT_NEAR("mat size(0)", (double)tensor_size(mat, 0), 2.0, 1e-10);
    ASSERT_NEAR("mat size(1)", (double)tensor_size(mat, 1), 3.0, 1e-10);
    ASSERT_NEAR("mat numel", (double)tensor_numel(mat), 6.0, 1e-10);

    double vec_data[] = {1.0, 0.0, -1.0};
    int vec_shape[] = {3};
    TensorHandle vec = tensor_create(vec_data, vec_shape, 1, 0);

    /* mv: [2,3] x [3] = [2]: [1*1+2*0+3*(-1), 4*1+5*0+6*(-1)] = [-2, -2] */
    TensorHandle mv = tensor_mv(mat, vec);
    ASSERT_NEAR("mv[0]", tensor_item(tensor_select(mv, 0, 0)), -2.0, 1e-10);
    ASSERT_NEAR("mv[1]", tensor_item(tensor_select(mv, 0, 1)), -2.0, 1e-10);

    tensor_free(mat); tensor_free(vec); tensor_free(mv);
}

static void test_dot(void) {
    printf("\n--- Dot product ---\n");
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    int shape[] = {3};
    TensorHandle va = tensor_create(a, shape, 1, 0);
    TensorHandle vb = tensor_create(b, shape, 1, 0);
    TensorHandle d = tensor_dot(va, vb);
    ASSERT_NEAR("dot([1,2,3],[4,5,6])", tensor_item(d), 32.0, 1e-10);
    tensor_free(va); tensor_free(vb); tensor_free(d);
}

static void test_softmax(void) {
    printf("\n--- Softmax ---\n");
    double data[] = {1.0, 2.0, 3.0};
    int shape[] = {3};
    TensorHandle v = tensor_create(data, shape, 1, 0);
    TensorHandle sm = tensor_softmax(v, 0);
    TensorHandle s = tensor_sum(sm);
    ASSERT_NEAR("softmax sums to 1", tensor_item(s), 1.0, 1e-6);
    tensor_free(v); tensor_free(sm); tensor_free(s);
}

static void test_outer(void) {
    printf("\n--- Outer product ---\n");
    double a[] = {1.0, 2.0};
    double b[] = {3.0, 4.0, 5.0};
    int sa[] = {2};
    int sb[] = {3};
    TensorHandle va = tensor_create(a, sa, 1, 0);
    TensorHandle vb = tensor_create(b, sb, 1, 0);
    TensorHandle o = tensor_outer(va, vb);
    /* [[3,4,5],[6,8,10]] */
    ASSERT_NEAR("outer dim", (double)tensor_dim(o), 2.0, 1e-10);
    ASSERT_NEAR("outer[0,0]", tensor_item_2d(o, 0, 0), 3.0, 1e-10);
    ASSERT_NEAR("outer[1,2]", tensor_item_2d(o, 1, 2), 10.0, 1e-10);
    tensor_free(va); tensor_free(vb); tensor_free(o);
}

/* ================================================================
   T4: Fused tensor ops with metadata backward
   ================================================================ */

static void test_fused_mv_backward(void) {
    printf("\n--- Fused MV backward (consolidated weight tensor) ---\n");
    param_clear();

    /* W = [[1,2,3],[4,5,6]], x = [1, 0, -1] */
    /* y = W @ x = [-2, -2], loss = sum(y) = -4 */
    /* d_W[i,j] = d_loss/d_W[i,j] = x[j] (since d_sum/d_y = [1,1]) */
    /* So grad_W = [[1,0,-1],[1,0,-1]] */
    /* d_x[j] = sum_i W[i,j] = [5, 7, 9] */

    double wdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    TensorHandle W = tensor_create_param_2d(2, 3, wdata);
    param_register("W", W);

    double xdata[] = {1.0, 0.0, -1.0};
    TensorHandle x = tensor_create_param_1d(3, xdata);
    param_register("x", x);

    TensorHandle y = tensor_mv(W, x);
    TensorHandle loss = tensor_sum(y);

    ASSERT_NEAR("mv loss", tensor_item(loss), -4.0, 1e-10);
    ASSERT_TRUE("loss requires_grad", tensor_requires_grad(loss));

    tensor_backward(loss);

    /* Check W gradients: grad_W[i,j] = x[j] */
    TensorHandle gW = tensor_grad(W);
    ASSERT_TRUE("W grad defined", gW != NULL);
    if (gW) {
        /* W is [2,3] param — grad should also be [2,3] */
        /* grad_W[0,0] = x[0] = 1, grad_W[0,1] = x[1] = 0, grad_W[0,2] = x[2] = -1 */
        ASSERT_NEAR("grad_W[0,0]", tensor_item_2d(W, 0, 0) != 0 ? param_grad_item(0) : -999, 1.0, 1e-6);
    }

    /* Check x gradients: grad_x[j] = sum_i W[i,j] */
    /* grad_x[0] = 1+4 = 5, grad_x[1] = 2+5 = 7, grad_x[2] = 3+6 = 9 */
    /* But x is a 1D param with 3 elements — param_grad_item reads element 0 */
    /* We need to check grad on the Tensor* directly */
    TensorHandle gx = tensor_grad(x);
    ASSERT_TRUE("x grad defined", gx != NULL);

    param_clear();
}

static void test_fused_mv_optimizer(void) {
    printf("\n--- Fused MV with optimizer (2 epochs) ---\n");
    param_clear();

    double wdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    TensorHandle W = tensor_create_param_2d(2, 3, wdata);
    param_register("W", W);

    double xdata[] = {1.0, 0.0, -1.0};
    int xshape[] = {3};
    TensorHandle x = tensor_create(xdata, xshape, 1, 0);  /* not a param */

    OptimizerHandle sgd = optimizer_create_sgd(0.1);

    double prev_loss = 1e10;
    for (int ep = 0; ep < 5; ep++) {
        optimizer_zero_grad(sgd);
        TensorHandle y = tensor_mv(W, x);
        TensorHandle loss = tensor_sum(y);
        double lv = tensor_item(loss);
        if (ep > 0) {
            ASSERT_TRUE("loss decreasing", lv < prev_loss + 0.01);
        }
        prev_loss = lv;
        tensor_backward(loss);
        optimizer_step(sgd);
    }
    ASSERT_TRUE("fused MV trains", prev_loss < -4.0);

    optimizer_free(sgd);
    param_clear();
}

/* ================================================================
   T5: LSTM-like gradient chain
   Mimics: param → MV → LSTM_GATES → SELECT → scalar loss → backward
   ================================================================ */

static void test_lstm_gradient_chain(void) {
    printf("\n--- LSTM gradient chain ---\n");
    param_clear();

    /* Create weight param [4, 2] — 4*o x i where o=1, i=2 */
    double w_data[] = {0.1, 0.2,   /* input gate row */
                       0.3, 0.4,   /* forget gate row */
                       0.5, 0.6,   /* cell gate row */
                       0.7, 0.8};  /* output gate row */
    TensorHandle w = tensor_create_param_2d(4, 2, w_data);
    param_register("w", w);

    /* Create bias param [4] */
    double b_data[] = {0.0, 1.0, 0.0, 0.0};  /* forget bias = 1 */
    TensorHandle b = tensor_create_param_1d(4, b_data);
    param_register("b", b);

    /* Input [2] — not a param, requires_grad=0 */
    double x_data[] = {1.0, 0.5};
    int x_shape[] = {2};
    TensorHandle x = tensor_create(x_data, x_shape, 1, 0);

    /* Prev cell [1] — requires_grad=0 (like initial state) */
    double c_data[] = {0.0};
    int c_shape[] = {1};
    TensorHandle prev_cell = tensor_create(c_data, c_shape, 1, 0);

    /* Forward: combined = w @ x + b */
    TensorHandle mv_result = tensor_mv(w, x);
    TensorHandle combined = tensor_add(mv_result, b);

    printf("combined requires_grad: %d\n", tensor_requires_grad(combined));
    ASSERT_TRUE("combined has rg=1", tensor_requires_grad(combined) == 1);

    /* LSTM gates */
    TensorPair* pair = tensor_lstm_gates_pair(combined, prev_cell, 1);
    TensorHandle hidden = tensor_pair_first(pair);
    TensorHandle cell = tensor_pair_second(pair);

    printf("hidden requires_grad: %d\n", tensor_requires_grad(hidden));
    ASSERT_TRUE("hidden has rg=1", tensor_requires_grad(hidden) == 1);

    /* SELECT: extract hidden scalar (like tensorToScalars) */
    TensorHandle h_scalar = tensor_select(hidden, 0, 0);
    printf("h_scalar requires_grad: %d, value: %f\n",
           tensor_requires_grad(h_scalar), tensor_item(h_scalar));
    ASSERT_TRUE("h_scalar has rg=1", tensor_requires_grad(h_scalar) == 1);

    /* Scalar loss: simple (h - target)^2 */
    TensorHandle target = tensor_create_scalar(0.5, 0);
    TensorHandle diff = tensor_sub(h_scalar, target);
    TensorHandle loss = tensor_mul(diff, diff);

    printf("loss value: %f, requires_grad: %d\n",
           tensor_item(loss), tensor_requires_grad(loss));

    /* Backward */
    tensor_backward(loss);

    /* Check param gradients via param_grad_item */
    double gw0 = param_grad_item(0);
    double gb0 = param_grad_item(1);
    printf("param grad w[0]=%f, b[0]=%f\n", gw0, gb0);
    /* At least some gradient should be non-zero */
    int w_has_grad = 0;
    for (int i = 0; i < 8; i++) {
        /* param_grad_item reads tensor->grad[0] only for scalars.
           For multi-element params we need more... but let's see if the pointer is non-null */
    }
    /* Use the debug print approach: check if backward reached MV */
    ASSERT_TRUE("w param has gradient (grad_item)", gw0 != 0.0 || gb0 != 0.0);

    param_clear();
    /* pair is arena-allocated, freed by arena_reset */
}

/* T5b: LSTM chain with STACK (mimics vecStackTensor round-trip) */
static void test_lstm_select_stack_chain(void) {
    printf("\n--- LSTM SELECT → STACK → MV chain ---\n");
    param_clear();

    /* Param: linear weight [1, 2] */
    double lw_data[] = {0.3, 0.7};
    TensorHandle lw = tensor_create_param_2d(1, 2, lw_data);
    param_register("lw", lw);

    /* Create a hidden vector [2] with requires_grad (like LSTM output) */
    double h_data[] = {0.4, 0.6};
    int h_shape[] = {2};
    TensorHandle hidden = tensor_create(h_data, h_shape, 1, 1);

    /* SELECT each element (like tensorToScalars) */
    TensorHandle s0 = tensor_select(hidden, 0, 0);
    TensorHandle s1 = tensor_select(hidden, 0, 1);

    printf("s0 rg=%d, s1 rg=%d\n", tensor_requires_grad(s0), tensor_requires_grad(s1));

    /* STACK them back (like vecStackTensor) */
    TensorHandle* ptr_arr = tensor_ptr_array_alloc(2);
    tensor_ptr_array_set(ptr_arr, 0, s0);
    tensor_ptr_array_set(ptr_arr, 1, s1);
    TensorHandle stacked = tensor_stack_from_array(ptr_arr, 2, 0);

    printf("stacked rg=%d, numel=%d\n",
           tensor_requires_grad(stacked), tensor_numel(stacked));
    ASSERT_TRUE("stacked has rg=1", tensor_requires_grad(stacked) == 1);

    /* MV with linear weight */
    TensorHandle mv_result = tensor_mv(lw, stacked);
    printf("mv_result rg=%d, value=%f\n",
           tensor_requires_grad(mv_result), tensor_item(mv_result));

    /* SELECT output (like tensorToScalars for output size 1) */
    TensorHandle out = tensor_select(mv_result, 0, 0);

    /* Scalar loss: (out - 1.0)^2 */
    TensorHandle target = tensor_create_scalar(1.0, 0);
    TensorHandle diff = tensor_sub(out, target);
    TensorHandle loss = tensor_mul(diff, diff);

    printf("loss=%f\n", tensor_item(loss));

    /* Backward */
    tensor_backward(loss);

    /* Check linear weight gradient */
    double glw = param_grad_item(0);
    printf("lw grad_item(0)=%f\n", glw);
    ASSERT_TRUE("linear weight has non-zero gradient", glw != 0.0);

    param_clear();
}

/* ================================================================
   T6: NTM read head gradient check (finite differences)
   ================================================================ */

/* Helper: run fused NTM read head forward, sum outputs, return scalar loss */
static double ntm_read_forward_loss(
    double* mem_data, int n, int w,
    double* prev_w_data, double* key_data,
    double beta_v, double g_v, double gamma_v,
    double* shift_data, int k)
{
    param_clear();
    int mem_shape[] = {n, w};
    TensorHandle mem = tensor_create(mem_data, mem_shape, 2, 1);
    int pw_shape[] = {n};
    TensorHandle pw = tensor_create(prev_w_data, pw_shape, 1, 1);
    int key_shape[] = {w};
    TensorHandle key = tensor_create(key_data, key_shape, 1, 1);
    TensorHandle beta = tensor_create_scalar(beta_v, 1);
    TensorHandle g = tensor_create_scalar(g_v, 1);
    TensorHandle gamma = tensor_create_scalar(gamma_v, 1);
    int s_shape[] = {k};
    TensorHandle shift = tensor_create(shift_data, s_shape, 1, 1);

    TensorPair* pair = tensor_ntm_read_head(mem, pw, key, beta, g, gamma, shift);
    TensorHandle focused = tensor_pair_first(pair);
    TensorHandle read_out = tensor_pair_second(pair);

    /* loss = sum(focused) + sum(read_out) */
    TensorHandle loss = tensor_add(tensor_sum(focused), tensor_sum(read_out));
    double val = tensor_item(loss);
    /* pair is arena-allocated */
    return val;
}

static void test_ntm_read_head_grad(void) {
    printf("\n--- NTM read head gradient check ---\n");
    param_clear();

    int n = 4, w = 3, k = 3;
    double mem[] = {0.1, 0.2, 0.3,
                    0.4, 0.5, 0.6,
                    0.7, 0.8, 0.9,
                    0.01, 0.02, 0.03};
    double prev_w[] = {0.25, 0.25, 0.25, 0.25};
    double key_data[] = {0.5, 0.3, 0.1};
    double beta_v = 1.5, g_v = 0.7, gamma_v = 1.2;
    double shift_data[] = {0.1, 0.8, 0.1};

    /* Analytical gradient via backward */
    int mem_shape[] = {n, w};
    TensorHandle memT = tensor_create(mem, mem_shape, 2, 1);
    param_register("mem", memT);
    int pw_shape[] = {n};
    TensorHandle pwT = tensor_create(prev_w, pw_shape, 1, 1);
    param_register("pw", pwT);
    int key_shape[] = {w};
    TensorHandle keyT = tensor_create(key_data, key_shape, 1, 1);
    param_register("key", keyT);
    TensorHandle betaT = tensor_create_scalar(beta_v, 1);
    param_register("beta", betaT);
    TensorHandle gT = tensor_create_scalar(g_v, 1);
    param_register("g", gT);
    TensorHandle gammaT = tensor_create_scalar(gamma_v, 1);
    param_register("gamma", gammaT);
    int s_shape[] = {k};
    TensorHandle shiftT = tensor_create(shift_data, s_shape, 1, 1);
    param_register("shift", shiftT);

    TensorPair* pair = tensor_ntm_read_head(memT, pwT, keyT, betaT, gT, gammaT, shiftT);
    TensorHandle focused = tensor_pair_first(pair);
    TensorHandle read_out = tensor_pair_second(pair);

    TensorHandle loss = tensor_add(tensor_sum(focused), tensor_sum(read_out));
    tensor_backward(loss);

    /* Params: 0=mem, 1=pw, 2=key, 3=beta, 4=g, 5=gamma, 6=shift */

    /* Check key gradient via finite differences */
    double eps = 1e-5;
    double key_copy[3];
    int key_ok = 1;
    for (int j = 0; j < w; j++) {
        memcpy(key_copy, key_data, w * sizeof(double));
        key_copy[j] += eps;
        double f_plus = ntm_read_forward_loss(mem, n, w, prev_w, key_copy, beta_v, g_v, gamma_v, shift_data, k);
        key_copy[j] = key_data[j] - eps;
        double f_minus = ntm_read_forward_loss(mem, n, w, prev_w, key_copy, beta_v, g_v, gamma_v, shift_data, k);
        double fd_grad = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(2, j);
        double err = fabs(fd_grad - analytic);
        printf("  key[%d]: fd=%f analytic=%f err=%e\n", j, fd_grad, analytic, err);
        if (err > 1e-3) key_ok = 0;
    }
    ASSERT_TRUE("key gradient matches finite diff", key_ok);

    /* Check beta gradient */
    {
        double f_plus = ntm_read_forward_loss(mem, n, w, prev_w, key_data, beta_v + eps, g_v, gamma_v, shift_data, k);
        double f_minus = ntm_read_forward_loss(mem, n, w, prev_w, key_data, beta_v - eps, g_v, gamma_v, shift_data, k);
        double fd_grad = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(3, 0);
        printf("  beta: fd=%f analytic=%f err=%e\n", fd_grad, analytic, fabs(fd_grad - analytic));
        ASSERT_NEAR("beta gradient", analytic, fd_grad, 1e-3);
    }

    /* Check g gradient */
    {
        double f_plus = ntm_read_forward_loss(mem, n, w, prev_w, key_data, beta_v, g_v + eps, gamma_v, shift_data, k);
        double f_minus = ntm_read_forward_loss(mem, n, w, prev_w, key_data, beta_v, g_v - eps, gamma_v, shift_data, k);
        double fd_grad = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(4, 0);
        printf("  g: fd=%f analytic=%f err=%e\n", fd_grad, analytic, fabs(fd_grad - analytic));
        ASSERT_NEAR("g gradient", analytic, fd_grad, 1e-3);
    }

    /* Check memory gradient (first element) */
    {
        double mem_copy[12];
        memcpy(mem_copy, mem, 12 * sizeof(double));
        mem_copy[0] += eps;
        double f_plus = ntm_read_forward_loss(mem_copy, n, w, prev_w, key_data, beta_v, g_v, gamma_v, shift_data, k);
        mem_copy[0] = mem[0] - eps;
        double f_minus = ntm_read_forward_loss(mem_copy, n, w, prev_w, key_data, beta_v, g_v, gamma_v, shift_data, k);
        double fd_grad = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(0, 0);
        printf("  mem[0]: fd=%f analytic=%f err=%e\n", fd_grad, analytic, fabs(fd_grad - analytic));
        ASSERT_NEAR("mem[0] gradient", analytic, fd_grad, 1e-3);
    }

    param_clear();
    /* pair is arena-allocated */
}

/* ================================================================
   T7: Transformer ops (matrix multiply, transpose, softmax_2d)
   ================================================================ */

static void test_mm_and_transpose(void) {
    printf("\n--- Matrix multiply + transpose ---\n");
    param_clear();

    /* A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2) */
    double a_data[] = {1,2,3, 4,5,6};
    double b_data[] = {7,8, 9,10, 11,12};
    int a_shape[] = {2, 3};
    int b_shape[] = {3, 2};
    TensorHandle a = tensor_create(a_data, a_shape, 2, 0);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 0);

    /* C = A @ B = [[58,64],[139,154]] */
    TensorHandle c = tensor_mm(a, b);
    ASSERT_NEAR("mm[0,0]", tensor_item_2d(c, 0, 0), 58.0, 1e-10);
    ASSERT_NEAR("mm[0,1]", tensor_item_2d(c, 0, 1), 64.0, 1e-10);
    ASSERT_NEAR("mm[1,0]", tensor_item_2d(c, 1, 0), 139.0, 1e-10);
    ASSERT_NEAR("mm[1,1]", tensor_item_2d(c, 1, 1), 154.0, 1e-10);

    /* Transpose: A^T should be [[1,4],[2,5],[3,6]] (3x2) */
    TensorHandle at = tensor_transpose_2d(a);
    ASSERT_NEAR("transpose[0,0]", tensor_item_2d(at, 0, 0), 1.0, 1e-10);
    ASSERT_NEAR("transpose[0,1]", tensor_item_2d(at, 0, 1), 4.0, 1e-10);
    ASSERT_NEAR("transpose[2,1]", tensor_item_2d(at, 2, 1), 6.0, 1e-10);
}

static void test_softmax_2d(void) {
    printf("\n--- Row-wise softmax 2D ---\n");
    /* 2x3 matrix, each row should sum to 1 */
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[] = {2, 3};
    TensorHandle t = tensor_create(data, shape, 2, 0);
    TensorHandle s = tensor_softmax_2d(t);

    double row0_sum = tensor_item_2d(s, 0, 0) + tensor_item_2d(s, 0, 1) + tensor_item_2d(s, 0, 2);
    double row1_sum = tensor_item_2d(s, 1, 0) + tensor_item_2d(s, 1, 1) + tensor_item_2d(s, 1, 2);
    ASSERT_NEAR("softmax_2d row0 sum", row0_sum, 1.0, 1e-10);
    ASSERT_NEAR("softmax_2d row1 sum", row1_sum, 1.0, 1e-10);
    /* Max element in each row should have highest probability */
    ASSERT_TRUE("softmax_2d row0 max", tensor_item_2d(s, 0, 2) > tensor_item_2d(s, 0, 0));
    ASSERT_TRUE("softmax_2d row1 max", tensor_item_2d(s, 1, 2) > tensor_item_2d(s, 1, 0));
}

static void test_mm_backward(void) {
    printf("\n--- Matrix multiply backward (finite diff) ---\n");
    param_clear();

    double a_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    double b_data[] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    int a_shape[] = {2, 3};
    int b_shape[] = {3, 2};

    /* Analytical gradient */
    TensorHandle a = tensor_create(a_data, a_shape, 2, 1);
    param_register("a", a);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 1);
    param_register("b", b);

    TensorHandle c = tensor_mm(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);

    /* Finite diff check for a[0,0] */
    double eps = 1e-5;
    double a_copy[6];
    memcpy(a_copy, a_data, 6 * sizeof(double));
    a_copy[0] += eps;
    {
        param_clear();
        TensorHandle a2 = tensor_create(a_copy, a_shape, 2, 0);
        TensorHandle b2 = tensor_create(b_data, b_shape, 2, 0);
        double f_plus = tensor_item(tensor_sum(tensor_mm(a2, b2)));
        a_copy[0] = a_data[0] - eps;
        TensorHandle a3 = tensor_create(a_copy, a_shape, 2, 0);
        TensorHandle b3 = tensor_create(b_data, b_shape, 2, 0);
        double f_minus = tensor_item(tensor_sum(tensor_mm(a3, b3)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(0, 0);
        printf("  a[0,0]: fd=%f analytic=%f err=%e\n", fd, analytic, fabs(fd - analytic));
        ASSERT_NEAR("mm grad a[0,0]", analytic, fd, 1e-3);
    }

    param_clear();
}

static void test_layer_norm_2d(void) {
    printf("\n--- Layer norm 2D forward ---\n");
    /* 2x3 matrix */
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double gamma_data[] = {1.0, 1.0, 1.0};
    double beta_data[] = {0.0, 0.0, 0.0};
    int shape[] = {2, 3};
    int gamma_shape[] = {3};

    TensorHandle t = tensor_create(data, shape, 2, 0);
    TensorHandle gamma = tensor_create(gamma_data, gamma_shape, 1, 0);
    TensorHandle beta = tensor_create(beta_data, gamma_shape, 1, 0);

    TensorHandle out = tensor_layer_norm_2d(t, gamma, beta, 1e-5);

    /* With gamma=1, beta=0, output should be standardized per row.
       Row 0: [1,2,3] mean=2 var=2/3 std~0.8165
       x_hat = [-1.2247, 0, 1.2247]
       Row 1: [4,5,6] mean=5 var=2/3 std~0.8165
       x_hat = [-1.2247, 0, 1.2247] */
    double std_val = sqrt(2.0/3.0 + 1e-5);
    ASSERT_NEAR("ln row0[0]", tensor_item_2d(out, 0, 0), -1.0/std_val, 1e-3);
    ASSERT_NEAR("ln row0[1]", tensor_item_2d(out, 0, 1), 0.0, 1e-3);
    ASSERT_NEAR("ln row0[2]", tensor_item_2d(out, 0, 2), 1.0/std_val, 1e-3);
    ASSERT_NEAR("ln row1[0]", tensor_item_2d(out, 1, 0), -1.0/std_val, 1e-3);
    ASSERT_NEAR("ln row1[1]", tensor_item_2d(out, 1, 1), 0.0, 1e-3);
    ASSERT_NEAR("ln row1[2]", tensor_item_2d(out, 1, 2), 1.0/std_val, 1e-3);

    /* With non-trivial gamma and beta */
    double gamma2[] = {2.0, 0.5, 1.0};
    double beta2[] = {1.0, -1.0, 0.5};
    TensorHandle gamma2h = tensor_create(gamma2, gamma_shape, 1, 0);
    TensorHandle beta2h = tensor_create(beta2, gamma_shape, 1, 0);
    TensorHandle out2 = tensor_layer_norm_2d(t, gamma2h, beta2h, 1e-5);
    /* Row 0: x_hat = [-1.2247, 0, 1.2247]
       y[0,0] = 2.0*(-1.2247) + 1.0 = -1.4494
       y[0,1] = 0.5*0 + (-1.0) = -1.0
       y[0,2] = 1.0*1.2247 + 0.5 = 1.7247 */
    double xh = 1.0 / std_val;
    ASSERT_NEAR("ln2 [0,0]", tensor_item_2d(out2, 0, 0), 2.0*(-xh) + 1.0, 1e-3);
    ASSERT_NEAR("ln2 [0,1]", tensor_item_2d(out2, 0, 1), 0.5*0.0 + (-1.0), 1e-3);
    ASSERT_NEAR("ln2 [0,2]", tensor_item_2d(out2, 0, 2), 1.0*xh + 0.5, 1e-3);
}

static void test_layer_norm_2d_backward(void) {
    printf("\n--- Layer norm 2D backward (finite diff) ---\n");
    param_clear();

    double data[] = {0.5, -0.3, 1.2, -0.7, 0.8, 0.1};
    double gamma_data[] = {0.8, 1.2, 0.5};
    double beta_data[] = {0.1, -0.2, 0.3};
    int shape[] = {2, 3};
    int gamma_shape[] = {3};

    /* Analytical gradient */
    TensorHandle t = tensor_create(data, shape, 2, 1);
    param_register("input", t);
    TensorHandle gamma = tensor_create(gamma_data, gamma_shape, 1, 1);
    param_register("gamma", gamma);
    TensorHandle beta = tensor_create(beta_data, gamma_shape, 1, 1);
    param_register("beta", beta);

    TensorHandle out = tensor_layer_norm_2d(t, gamma, beta, 1e-5);
    TensorHandle loss = tensor_sum(out);
    tensor_backward(loss);

    /* Finite diff check for input, gamma, beta */
    double eps = 1e-5;

    /* Check input[0,0] */
    {
        double d_copy[6]; memcpy(d_copy, data, sizeof(data));
        param_clear();
        d_copy[0] = data[0] + eps;
        TensorHandle t2 = tensor_create(d_copy, shape, 2, 0);
        TensorHandle g2 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b2 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_layer_norm_2d(t2, g2, b2, 1e-5)));
        d_copy[0] = data[0] - eps;
        TensorHandle t3 = tensor_create(d_copy, shape, 2, 0);
        TensorHandle g3 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b3 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_layer_norm_2d(t3, g3, b3, 1e-5)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(0, 0);
        printf("  input[0,0]: fd=%f analytic=%f err=%e\n", fd, analytic, fabs(fd - analytic));
        ASSERT_NEAR("ln grad input[0,0]", analytic, fd, 1e-3);
    }

    /* Check gamma[0] */
    {
        double g_copy[3]; memcpy(g_copy, gamma_data, sizeof(gamma_data));
        param_clear();
        g_copy[0] = gamma_data[0] + eps;
        TensorHandle t2 = tensor_create(data, shape, 2, 0);
        TensorHandle g2 = tensor_create(g_copy, gamma_shape, 1, 0);
        TensorHandle b2 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_layer_norm_2d(t2, g2, b2, 1e-5)));
        g_copy[0] = gamma_data[0] - eps;
        TensorHandle t3 = tensor_create(data, shape, 2, 0);
        TensorHandle g3 = tensor_create(g_copy, gamma_shape, 1, 0);
        TensorHandle b3 = tensor_create(beta_data, gamma_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_layer_norm_2d(t3, g3, b3, 1e-5)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(1, 0);
        printf("  gamma[0]: fd=%f analytic=%f err=%e\n", fd, analytic, fabs(fd - analytic));
        ASSERT_NEAR("ln grad gamma[0]", analytic, fd, 1e-3);
    }

    /* Check beta[1] */
    {
        double b_copy[3]; memcpy(b_copy, beta_data, sizeof(beta_data));
        param_clear();
        b_copy[1] = beta_data[1] + eps;
        TensorHandle t2 = tensor_create(data, shape, 2, 0);
        TensorHandle g2 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b2 = tensor_create(b_copy, gamma_shape, 1, 0);
        double f_plus = tensor_item(tensor_sum(tensor_layer_norm_2d(t2, g2, b2, 1e-5)));
        b_copy[1] = beta_data[1] - eps;
        TensorHandle t3 = tensor_create(data, shape, 2, 0);
        TensorHandle g3 = tensor_create(gamma_data, gamma_shape, 1, 0);
        TensorHandle b3 = tensor_create(b_copy, gamma_shape, 1, 0);
        double f_minus = tensor_item(tensor_sum(tensor_layer_norm_2d(t3, g3, b3, 1e-5)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(2, 1);
        printf("  beta[1]: fd=%f analytic=%f err=%e\n", fd, analytic, fabs(fd - analytic));
        ASSERT_NEAR("ln grad beta[1]", analytic, fd, 1e-3);
    }

    param_clear();
}

static void test_bmm_forward(void) {
    printf("\n--- Batched matrix multiply forward ---\n");
    /* a = [2, 2, 3], b = [3, 2] => result = [2, 2, 2] */
    double a_data[] = {
        /* batch 0: [[1,2,3],[4,5,6]] */
        1, 2, 3, 4, 5, 6,
        /* batch 1: [[7,8,9],[10,11,12]] */
        7, 8, 9, 10, 11, 12
    };
    double b_data[] = {1, 0, 0, 1, 1, 1};  /* [[1,0],[0,1],[1,1]] */
    int a_shape[] = {2, 2, 3};
    int b_shape[] = {3, 2};

    TensorHandle a = tensor_create(a_data, a_shape, 3, 0);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 0);
    TensorHandle c = tensor_bmm(a, b);

    /* Read results into flat buffer for rank-3 indexing */
    double out[8];
    tensor_to_doubles(c, out);
    /* batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]] */
    ASSERT_NEAR("bmm[0,0,0]", out[0], 4.0, 1e-10);
    ASSERT_NEAR("bmm[0,0,1]", out[1], 5.0, 1e-10);
    ASSERT_NEAR("bmm[0,1,0]", out[2], 10.0, 1e-10);
    ASSERT_NEAR("bmm[0,1,1]", out[3], 11.0, 1e-10);
    /* batch 1: [[7,8,9],[10,11,12]] @ [[1,0],[0,1],[1,1]] = [[16,17],[22,23]] */
    ASSERT_NEAR("bmm[1,0,0]", out[4], 16.0, 1e-10);
    ASSERT_NEAR("bmm[1,0,1]", out[5], 17.0, 1e-10);
    ASSERT_NEAR("bmm[1,1,0]", out[6], 22.0, 1e-10);
    ASSERT_NEAR("bmm[1,1,1]", out[7], 23.0, 1e-10);
}

static void test_bmm_backward(void) {
    printf("\n--- Batched matrix multiply backward (finite diff) ---\n");
    param_clear();

    double a_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    double b_data[] = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    int a_shape[] = {2, 2, 3};
    int b_shape[] = {3, 2};

    TensorHandle a = tensor_create(a_data, a_shape, 3, 1);
    param_register("a", a);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 1);
    param_register("b", b);

    TensorHandle c = tensor_bmm(a, b);
    TensorHandle loss = tensor_sum(c);
    tensor_backward(loss);

    /* Finite diff check for a[0] (first element of batch 0) */
    double eps = 1e-5;
    {
        double a_copy[12]; memcpy(a_copy, a_data, sizeof(a_data));
        param_clear();
        a_copy[0] = a_data[0] + eps;
        TensorHandle a2 = tensor_create(a_copy, a_shape, 3, 0);
        TensorHandle b2 = tensor_create(b_data, b_shape, 2, 0);
        double f_plus = tensor_item(tensor_sum(tensor_bmm(a2, b2)));
        a_copy[0] = a_data[0] - eps;
        TensorHandle a3 = tensor_create(a_copy, a_shape, 3, 0);
        TensorHandle b3 = tensor_create(b_data, b_shape, 2, 0);
        double f_minus = tensor_item(tensor_sum(tensor_bmm(a3, b3)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(0, 0);
        printf("  a[0]: fd=%f analytic=%f err=%e\n", fd, analytic, fabs(fd - analytic));
        ASSERT_NEAR("bmm grad a[0]", analytic, fd, 1e-3);
    }

    /* Finite diff check for b[0] (weight grad, accumulated across batch) */
    {
        double b_copy[6]; memcpy(b_copy, b_data, sizeof(b_data));
        param_clear();
        b_copy[0] = b_data[0] + eps;
        TensorHandle a2 = tensor_create(a_data, a_shape, 3, 0);
        TensorHandle b2 = tensor_create(b_copy, b_shape, 2, 0);
        double f_plus = tensor_item(tensor_sum(tensor_bmm(a2, b2)));
        b_copy[0] = b_data[0] - eps;
        TensorHandle a3 = tensor_create(a_data, a_shape, 3, 0);
        TensorHandle b3 = tensor_create(b_copy, b_shape, 2, 0);
        double f_minus = tensor_item(tensor_sum(tensor_bmm(a3, b3)));
        double fd = (f_plus - f_minus) / (2 * eps);
        double analytic = param_grad_item_at(1, 0);
        printf("  b[0]: fd=%f analytic=%f err=%e\n", fd, analytic, fabs(fd - analytic));
        ASSERT_NEAR("bmm grad b[0]", analytic, fd, 1e-3);
    }

    param_clear();
}

/* ================================================================
   Main
   ================================================================ */

int main(void) {
    /* T1 */
    test_scalar_creation();
    test_arithmetic();

    /* T2 */
    test_autograd_basic();
    test_autograd_chain();
    test_autograd_exp();
    test_autograd_div();
    test_autograd_sqrt();
    test_autograd_native_sgd();

    /* T3 */
    test_multidim();
    test_dot();
    test_softmax();
    test_outer();

    /* T4: Fused ops */
    test_fused_mv_backward();
    test_fused_mv_optimizer();

    /* T5: LSTM gradient chain */
    test_lstm_gradient_chain();
    test_lstm_select_stack_chain();

    /* T6: NTM gradient check */
    test_ntm_read_head_grad();

    /* T7: Transformer ops */
    test_mm_and_transpose();
    test_softmax_2d();
    test_mm_backward();

    /* T8: Layer norm */
    test_layer_norm_2d();
    test_layer_norm_2d_backward();

    /* T9: Batched ops */
    test_bmm_forward();
    test_bmm_backward();

    /* Summary */
    printf("\n");
    if (failures == 0) {
        printf("All tape backend tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}

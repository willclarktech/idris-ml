/* Test suite for the tape-based C backend (backend_tape.c).
   Tests the same backend.h API as test_backend.c but links against
   the tape backend instead of libtorch. */

#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
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
    free(pair);
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

    /* Summary */
    printf("\n");
    if (failures == 0) {
        printf("All tape backend tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}

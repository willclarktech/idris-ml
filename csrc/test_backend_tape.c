/* Test suite for the tape-based C backend (backend_tape.c).
   Tests the same backend.h API as test_backend.c but links against
   the tape backend instead of libtorch. */

#include "backend.h"
#include <stdio.h>
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

    /* Summary */
    printf("\n");
    if (failures == 0) {
        printf("All tape backend tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}

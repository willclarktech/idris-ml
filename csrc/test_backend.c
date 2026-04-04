#include "backend.h"
#include <stdio.h>
#include <math.h>

#define ASSERT_NEAR(msg, got, expected, tol) do { \
    double _g = (got), _e = (expected); \
    if (fabs(_g - _e) > (tol)) { \
        printf("FAIL: %s: got %.6f, expected %.6f\n", msg, _g, _e); \
        failures++; \
    } else { \
        printf("ok: %s = %.6f\n", msg, _g); \
    } \
} while(0)

int main(void) {
    int failures = 0;

    /* --- Scalar creation and arithmetic --- */
    TensorHandle a = tensor_create_scalar(3.0, 0);
    TensorHandle b = tensor_create_scalar(4.0, 0);

    ASSERT_NEAR("scalar item", tensor_item(a), 3.0, 1e-10);
    ASSERT_NEAR("numel", (double)tensor_numel(a), 1.0, 1e-10);

    TensorHandle c = tensor_add(a, b);
    ASSERT_NEAR("3+4", tensor_item(c), 7.0, 1e-10);

    TensorHandle d = tensor_mul(a, b);
    ASSERT_NEAR("3*4", tensor_item(d), 12.0, 1e-10);

    TensorHandle e = tensor_sub(a, b);
    ASSERT_NEAR("3-4", tensor_item(e), -1.0, 1e-10);

    TensorHandle f = tensor_div(a, b);
    ASSERT_NEAR("3/4", tensor_item(f), 0.75, 1e-10);

    TensorHandle g = tensor_neg(a);
    ASSERT_NEAR("neg(3)", tensor_item(g), -3.0, 1e-10);

    TensorHandle h = tensor_exp(a);
    ASSERT_NEAR("exp(3)", tensor_item(h), exp(3.0), 1e-6);

    TensorHandle i = tensor_log(b);
    ASSERT_NEAR("log(4)", tensor_item(i), log(4.0), 1e-10);

    TensorHandle j = tensor_sqrt(b);
    ASSERT_NEAR("sqrt(4)", tensor_item(j), 2.0, 1e-10);

    TensorHandle k = tensor_sigmoid(a);
    ASSERT_NEAR("sigmoid(3)", tensor_item(k), 1.0/(1.0+exp(-3.0)), 1e-10);

    TensorHandle l = tensor_tanh(a);
    ASSERT_NEAR("tanh(3)", tensor_item(l), tanh(3.0), 1e-10);

    tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(d);
    tensor_free(e); tensor_free(f); tensor_free(g); tensor_free(h);
    tensor_free(i); tensor_free(j); tensor_free(k); tensor_free(l);

    /* --- Vector creation and matmul --- */
    double mat_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int mat_shape[] = {2, 3};
    TensorHandle mat = tensor_create(mat_data, mat_shape, 2, 0);
    ASSERT_NEAR("mat dim", (double)tensor_dim(mat), 2.0, 1e-10);
    ASSERT_NEAR("mat size(0)", (double)tensor_size(mat, 0), 2.0, 1e-10);
    ASSERT_NEAR("mat size(1)", (double)tensor_size(mat, 1), 3.0, 1e-10);

    double vec_data[] = {1.0, 0.0, -1.0};
    int vec_shape[] = {3};
    TensorHandle vec = tensor_create(vec_data, vec_shape, 1, 0);

    TensorHandle mv = tensor_mv(mat, vec);
    ASSERT_NEAR("mv[0]", tensor_item(tensor_select(mv, 0, 0)), -2.0, 1e-10);
    ASSERT_NEAR("mv[1]", tensor_item(tensor_select(mv, 0, 1)), -2.0, 1e-10);

    tensor_free(mat); tensor_free(vec); tensor_free(mv);

    /* --- Softmax --- */
    double sm_data[] = {1.0, 2.0, 3.0};
    int sm_shape[] = {3};
    TensorHandle sm_in = tensor_create(sm_data, sm_shape, 1, 0);
    TensorHandle sm_out = tensor_softmax(sm_in, 0);
    TensorHandle sm_sum = tensor_sum(sm_out);
    ASSERT_NEAR("softmax sums to 1", tensor_item(sm_sum), 1.0, 1e-6);

    tensor_free(sm_in); tensor_free(sm_out); tensor_free(sm_sum);

    /* --- Dot product --- */
    double dot_a[] = {1.0, 2.0, 3.0};
    double dot_b[] = {4.0, 5.0, 6.0};
    int dot_shape[] = {3};
    TensorHandle da = tensor_create(dot_a, dot_shape, 1, 0);
    TensorHandle db = tensor_create(dot_b, dot_shape, 1, 0);
    TensorHandle dp = tensor_dot(da, db);
    ASSERT_NEAR("dot([1,2,3],[4,5,6])", tensor_item(dp), 32.0, 1e-10);
    tensor_free(da); tensor_free(db); tensor_free(dp);

    /* --- Autograd --- */
    TensorHandle x = tensor_create_scalar(2.0, 1);
    TensorHandle y = tensor_create_scalar(3.0, 1);
    TensorHandle xy = tensor_mul(x, y);
    TensorHandle loss = tensor_sum(xy);
    tensor_backward(loss);

    TensorHandle gx = tensor_grad(x);
    TensorHandle gy = tensor_grad(y);
    ASSERT_NEAR("d(x*y)/dx at y=3", tensor_item(gx), 3.0, 1e-10);
    ASSERT_NEAR("d(x*y)/dy at x=2", tensor_item(gy), 2.0, 1e-10);

    tensor_free(gx); tensor_free(gy);
    tensor_free(x); tensor_free(y); tensor_free(xy); tensor_free(loss);

    /* --- Autograd chain: f = (a+b)^2 --- */
    TensorHandle p = tensor_create_scalar(1.0, 1);
    TensorHandle q = tensor_create_scalar(2.0, 1);
    TensorHandle s = tensor_add(p, q);        /* s = 3 */
    TensorHandle s2 = tensor_mul(s, s);       /* s2 = 9 */
    tensor_backward(s2);

    TensorHandle gp = tensor_grad(p);
    TensorHandle gq = tensor_grad(q);
    /* d/dp (p+q)^2 = 2(p+q) = 6 */
    ASSERT_NEAR("d((a+b)^2)/da", tensor_item(gp), 6.0, 1e-10);
    ASSERT_NEAR("d((a+b)^2)/db", tensor_item(gq), 6.0, 1e-10);

    tensor_free(gp); tensor_free(gq);
    tensor_free(p); tensor_free(q); tensor_free(s); tensor_free(s2);

    /* --- No-grad scope --- */
    tensor_no_grad_begin();
    TensorHandle ng = tensor_create_scalar(5.0, 0);
    TensorHandle ng2 = tensor_mul_scalar(ng, 2.0);
    ASSERT_NEAR("no_grad mul", tensor_item(ng2), 10.0, 1e-10);
    tensor_no_grad_end();
    tensor_free(ng); tensor_free(ng2);

    /* --- Summary --- */
    if (failures == 0) {
        printf("\nAll backend tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Pull in the implementation directly */
#include "tensor.c"

static int tests_passed = 0;
static int tests_failed = 0;

static void check(const char *name, int ok) {
  if (ok) {
    tests_passed++;
  } else {
    tests_failed++;
    printf("FAIL: %s\n", name);
  }
}

static void check_close(const char *name, double got, double expected,
                         double tol) {
  int ok = fabs(got - expected) < tol;
  if (!ok) {
    printf("FAIL: %s: got %.10f, expected %.10f\n", name, got, expected);
    tests_failed++;
  } else {
    tests_passed++;
  }
}


/* -------------------------------------------------------------------
   Forward tests
   ------------------------------------------------------------------- */

static void test_matvec_identity(void) {
  /* 2x2 identity matrix * [3, 7] = [3, 7] */
  double w[] = {1, 0, 0, 1};
  double x[] = {3, 7};
  double out[2] = {0};
  tensor_matvec(w, x, out, 2, 2);
  check_close("matvec_identity[0]", out[0], 3.0, 1e-12);
  check_close("matvec_identity[1]", out[1], 7.0, 1e-12);
}

static void test_matvec_general(void) {
  /* [[1, 2], [3, 4]] * [5, 6] = [17, 39] */
  double w[] = {1, 2, 3, 4};
  double x[] = {5, 6};
  double out[2] = {0};
  tensor_matvec(w, x, out, 2, 2);
  check_close("matvec_general[0]", out[0], 17.0, 1e-12);
  check_close("matvec_general[1]", out[1], 39.0, 1e-12);
}

static void test_matvec_nonsquare(void) {
  /* [[1, 2, 3], [4, 5, 6]] * [1, 0, -1] = [-2, -2] */
  double w[] = {1, 2, 3, 4, 5, 6};
  double x[] = {1, 0, -1};
  double out[2] = {0};
  tensor_matvec(w, x, out, 2, 3);
  check_close("matvec_nonsquare[0]", out[0], -2.0, 1e-12);
  check_close("matvec_nonsquare[1]", out[1], -2.0, 1e-12);
}

static void test_dot(void) {
  double a[] = {1, 2, 3};
  double b[] = {4, 5, 6};
  double r = tensor_dot(a, b, 3);
  check_close("dot", r, 32.0, 1e-12);
}

static void test_dot_orthogonal(void) {
  double a[] = {1, 0};
  double b[] = {0, 1};
  double r = tensor_dot(a, b, 2);
  check_close("dot_orthogonal", r, 0.0, 1e-12);
}


/* -------------------------------------------------------------------
   Backward tests
   ------------------------------------------------------------------- */

static void test_matvec_backward(void) {
  /*
   * W = [[1, 2], [3, 4]], x = [5, 6]
   * out = [17, 39]
   * dy = [1, 1]
   *
   * dW[i][j] = dy[i] * x[j]
   *   dW = [[5, 6], [5, 6]]
   *
   * dx[j] = sum_i dy[i] * W[i][j]
   *   dx = [1*1 + 1*3, 1*2 + 1*4] = [4, 6]
   */
  arena_reset();

  MatVecMeta *meta = matvec_meta_alloc(2, 2);
  double w_vals[] = {1, 2, 3, 4};
  double x_vals[] = {5, 6};
  memcpy(meta->w_vals, w_vals, 4 * sizeof(double));
  memcpy(meta->x_vals, x_vals, 2 * sizeof(double));

  /* Tape layout: indices 0-3 = weights, 4-5 = inputs, 6 = MatVecOp, 7-8 = outputs */
  int w_idx[] = {0, 1, 2, 3};
  int x_idx[] = {4, 5};
  memcpy(meta->w_tape_idx, w_idx, 4 * sizeof(int));
  memcpy(meta->x_tape_idx, x_idx, 2 * sizeof(int));
  meta->out_tape_start = 6;

  /* grad array: 9 entries */
  double grad[9] = {0};
  grad[7] = 1.0; /* dy[0] */
  grad[8] = 1.0; /* dy[1] */

  tensor_matvec_backward(grad, meta);

  check_close("matvec_bwd dW[0][0]", grad[0], 5.0, 1e-12);
  check_close("matvec_bwd dW[0][1]", grad[1], 6.0, 1e-12);
  check_close("matvec_bwd dW[1][0]", grad[2], 5.0, 1e-12);
  check_close("matvec_bwd dW[1][1]", grad[3], 6.0, 1e-12);
  check_close("matvec_bwd dx[0]", grad[4], 4.0, 1e-12);
  check_close("matvec_bwd dx[1]", grad[5], 6.0, 1e-12);
}

static void test_matvec_backward_scaled(void) {
  /*
   * Same as above but dy = [2, 0.5]
   * dW = [[10, 12], [2.5, 3]]
   * dx = [2*1 + 0.5*3, 2*2 + 0.5*4] = [3.5, 6]
   */
  arena_reset();

  MatVecMeta *meta = matvec_meta_alloc(2, 2);
  double w_vals[] = {1, 2, 3, 4};
  double x_vals[] = {5, 6};
  memcpy(meta->w_vals, w_vals, 4 * sizeof(double));
  memcpy(meta->x_vals, x_vals, 2 * sizeof(double));

  int w_idx[] = {0, 1, 2, 3};
  int x_idx[] = {4, 5};
  memcpy(meta->w_tape_idx, w_idx, 4 * sizeof(int));
  memcpy(meta->x_tape_idx, x_idx, 2 * sizeof(int));
  meta->out_tape_start = 6;

  double grad[9] = {0};
  grad[7] = 2.0;
  grad[8] = 0.5;

  tensor_matvec_backward(grad, meta);

  check_close("matvec_bwd_scaled dW[0][0]", grad[0], 10.0, 1e-12);
  check_close("matvec_bwd_scaled dW[0][1]", grad[1], 12.0, 1e-12);
  check_close("matvec_bwd_scaled dW[1][0]", grad[2], 2.5, 1e-12);
  check_close("matvec_bwd_scaled dW[1][1]", grad[3], 3.0, 1e-12);
  check_close("matvec_bwd_scaled dx[0]", grad[4], 3.5, 1e-12);
  check_close("matvec_bwd_scaled dx[1]", grad[5], 6.0, 1e-12);
}

static void test_dot_backward(void) {
  /*
   * a = [1, 2, 3], b = [4, 5, 6]
   * out = 32, dy = 1
   * da = dy * b = [4, 5, 6]
   * db = dy * a = [1, 2, 3]
   */
  arena_reset();

  DotMeta *meta = dot_meta_alloc(3);
  double a_vals[] = {1, 2, 3};
  double b_vals[] = {4, 5, 6};
  memcpy(meta->a_vals, a_vals, 3 * sizeof(double));
  memcpy(meta->b_vals, b_vals, 3 * sizeof(double));

  int a_idx[] = {0, 1, 2};
  int b_idx[] = {3, 4, 5};
  memcpy(meta->a_tape_idx, a_idx, 3 * sizeof(int));
  memcpy(meta->b_tape_idx, b_idx, 3 * sizeof(int));
  meta->out_tape_idx = 6;

  double grad[7] = {0};
  grad[6] = 1.0;

  tensor_dot_backward(grad, meta);

  check_close("dot_bwd da[0]", grad[0], 4.0, 1e-12);
  check_close("dot_bwd da[1]", grad[1], 5.0, 1e-12);
  check_close("dot_bwd da[2]", grad[2], 6.0, 1e-12);
  check_close("dot_bwd db[0]", grad[3], 1.0, 1e-12);
  check_close("dot_bwd db[1]", grad[4], 2.0, 1e-12);
  check_close("dot_bwd db[2]", grad[5], 3.0, 1e-12);
}


/* -------------------------------------------------------------------
   Softmax/LogSoftmax tests
   ------------------------------------------------------------------- */

static void test_softmax_forward(void) {
  /* x = [1, 2, 3]
   * softmax = [e^1, e^2, e^3] / (e^1 + e^2 + e^3)
   * sum should be 1.0 */
  arena_reset();
  SoftmaxMeta *meta = softmax_meta_alloc(3);
  meta->x_vals[0] = 1.0;
  meta->x_vals[1] = 2.0;
  meta->x_vals[2] = 3.0;

  double out[3] = {0};
  softmax_meta_compute(meta, out);

  double total = exp(1.0) + exp(2.0) + exp(3.0);
  check_close("softmax[0]", out[0], exp(1.0) / total, 1e-10);
  check_close("softmax[1]", out[1], exp(2.0) / total, 1e-10);
  check_close("softmax[2]", out[2], exp(3.0) / total, 1e-10);
  check_close("softmax_sum", out[0] + out[1] + out[2], 1.0, 1e-10);
}

static void test_logsoftmax_forward(void) {
  /* x = [1, 2, 3]
   * logsoftmax[i] = x[i] - log(sum(exp(x)))
   * exp(logsoftmax) should sum to 1.0 */
  arena_reset();
  SoftmaxMeta *meta = softmax_meta_alloc(3);
  meta->x_vals[0] = 1.0;
  meta->x_vals[1] = 2.0;
  meta->x_vals[2] = 3.0;

  double out[3] = {0};
  logsoftmax_meta_compute(meta, out);

  double logZ = log(exp(1.0) + exp(2.0) + exp(3.0));
  check_close("logsoftmax[0]", out[0], 1.0 - logZ, 1e-10);
  check_close("logsoftmax[1]", out[1], 2.0 - logZ, 1e-10);
  check_close("logsoftmax[2]", out[2], 3.0 - logZ, 1e-10);
  check_close("exp_logsoftmax_sum",
              exp(out[0]) + exp(out[1]) + exp(out[2]), 1.0, 1e-10);
}

static void test_softmax_backward(void) {
  /* x = [1, 2, 3], dy = [1, 0, 0]
   * s = softmax(x)
   * dot(dy, s) = s[0]
   * dx[j] = s[j] * (dy[j] - s[0])
   *   dx[0] = s[0] * (1 - s[0])
   *   dx[1] = s[1] * (0 - s[0]) = -s[1]*s[0]
   *   dx[2] = s[2] * (0 - s[0]) = -s[2]*s[0]
   */
  arena_reset();
  SoftmaxMeta *meta = softmax_meta_alloc(3);
  meta->x_vals[0] = 1.0;
  meta->x_vals[1] = 2.0;
  meta->x_vals[2] = 3.0;

  double out[3] = {0};
  softmax_meta_compute(meta, out);

  int x_idx[] = {0, 1, 2};
  memcpy(meta->x_tape_idx, x_idx, 3 * sizeof(int));
  meta->out_tape_start = 3;
  /* grad layout: [dx0, dx1, dx2, <op>, dy0, dy1, dy2] */
  double grad[7] = {0};
  grad[4] = 1.0; /* dy[0] */

  tensor_softmax_backward(grad, meta);

  double s0 = out[0], s1 = out[1], s2 = out[2];
  check_close("softmax_bwd dx[0]", grad[0], s0 * (1.0 - s0), 1e-10);
  check_close("softmax_bwd dx[1]", grad[1], -s1 * s0, 1e-10);
  check_close("softmax_bwd dx[2]", grad[2], -s2 * s0, 1e-10);
}

static void test_logsoftmax_backward(void) {
  /* x = [1, 2, 3], dy = [1, 0, 0]
   * logS = logsoftmax(x), s = exp(logS)
   * sum_dy = 1
   * dx[j] = dy[j] - s[j] * sum_dy
   *   dx[0] = 1 - s[0]
   *   dx[1] = 0 - s[1]
   *   dx[2] = 0 - s[2]
   */
  arena_reset();
  SoftmaxMeta *meta = softmax_meta_alloc(3);
  meta->x_vals[0] = 1.0;
  meta->x_vals[1] = 2.0;
  meta->x_vals[2] = 3.0;

  double out[3] = {0};
  logsoftmax_meta_compute(meta, out);

  int x_idx[] = {0, 1, 2};
  memcpy(meta->x_tape_idx, x_idx, 3 * sizeof(int));
  meta->out_tape_start = 3;
  double grad[7] = {0};
  grad[4] = 1.0; /* dy[0] */

  tensor_logsoftmax_backward(grad, meta);

  double s0 = exp(out[0]), s1 = exp(out[1]), s2 = exp(out[2]);
  check_close("logsoftmax_bwd dx[0]", grad[0], 1.0 - s0, 1e-10);
  check_close("logsoftmax_bwd dx[1]", grad[1], -s1, 1e-10);
  check_close("logsoftmax_bwd dx[2]", grad[2], -s2, 1e-10);
}


/* -------------------------------------------------------------------
   Arena tests
   ------------------------------------------------------------------- */

static void test_arena(void) {
  arena_reset();
  void *p1 = arena_alloc(16);
  void *p2 = arena_alloc(32);
  check("arena_non_null_1", p1 != NULL);
  check("arena_non_null_2", p2 != NULL);
  check("arena_no_overlap", (char *)p2 >= (char *)p1 + 16);
  arena_reset();
  void *p3 = arena_alloc(16);
  check("arena_reset_reuses", p3 == p1);
}


/* -------------------------------------------------------------------
   Main
   ------------------------------------------------------------------- */

int main(void) {
  printf("=== tensor.c tests ===\n");

  test_arena();
  test_matvec_identity();
  test_matvec_general();
  test_matvec_nonsquare();
  test_dot();
  test_dot_orthogonal();
  test_matvec_backward();
  test_matvec_backward_scaled();
  test_dot_backward();
  test_softmax_forward();
  test_logsoftmax_forward();
  test_softmax_backward();
  test_logsoftmax_backward();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}

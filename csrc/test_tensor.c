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
    printf("  PASS: %s\n", name);
  } else {
    tests_failed++;
    printf("  FAIL: %s\n", name);
  }
}

static void check_close(const char *name, double got, double expected,
                         double tol) {
  int ok = fabs(got - expected) < tol;
  if (!ok) {
    printf("  FAIL: %s: got %.10f, expected %.10f\n", name, got, expected);
    tests_failed++;
  } else {
    tests_passed++;
    printf("  PASS: %s\n", name);
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
  meta->out_tape_start = 7;

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
  meta->out_tape_start = 7;

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
  meta->out_tape_start = 4;
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
  meta->out_tape_start = 4;
  double grad[7] = {0};
  grad[4] = 1.0; /* dy[0] */

  tensor_logsoftmax_backward(grad, meta);

  double s0 = exp(out[0]), s1 = exp(out[1]), s2 = exp(out[2]);
  check_close("logsoftmax_bwd dx[0]", grad[0], 1.0 - s0, 1e-10);
  check_close("logsoftmax_bwd dx[1]", grad[1], -s1, 1e-10);
  check_close("logsoftmax_bwd dx[2]", grad[2], -s2, 1e-10);
}


/* -------------------------------------------------------------------
   BatchCosSim tests
   ------------------------------------------------------------------- */

static void test_batch_cossim_forward(void) {
  /* mem = [[1, 0], [0, 1], [1, 1]], key = [1, 0], beta = 10
   * cos(key, row0) = 1*1 / (1*1) = 1.0
   * cos(key, row1) = 0 / (1*1) = 0.0
   * cos(key, row2) = 1 / (1*sqrt(2)) = 1/sqrt(2)
   * out = [10, 0, 10/sqrt(2)] */
  arena_reset();
  BatchCosSimMeta *m = batch_cossim_meta_alloc(3, 2);
  double mem[] = {1, 0, 0, 1, 1, 1};
  double key[] = {1, 0};
  memcpy(m->mem_vals, mem, 6 * sizeof(double));
  memcpy(m->key_vals, key, 2 * sizeof(double));
  m->beta_val = 10.0;

  double out[3] = {0};
  batch_cossim_compute(m, out);

  check_close("batch_cossim[0]", out[0], 10.0, 1e-10);
  check_close("batch_cossim[1]", out[1], 0.0, 1e-10);
  check_close("batch_cossim[2]", out[2], 10.0 / sqrt(2.0), 1e-10);
}

static void test_batch_cossim_backward(void) {
  /* Numerical gradient check for batch cosine similarity.
   * mem = [[3, 1], [1, 2]], key = [2, 1], beta = 5 */
  arena_reset();
  double eps = 1e-5;

  double mem[] = {3, 1, 1, 2};
  double key[] = {2, 1};
  double beta = 5.0;
  int n = 2, w = 2;

  /* Compute forward */
  BatchCosSimMeta *m = batch_cossim_meta_alloc(n, w);
  memcpy(m->mem_vals, mem, 4 * sizeof(double));
  memcpy(m->key_vals, key, 2 * sizeof(double));
  m->beta_val = beta;
  double out[2];
  batch_cossim_compute(m, out);

  /* Tape layout: 0-3 = mem, 4-5 = key, 6 = beta, 7 = op, 8-9 = output */
  int mem_idx[] = {0, 1, 2, 3};
  int key_idx[] = {4, 5};
  memcpy(m->mem_tape_idx, mem_idx, 4 * sizeof(int));
  memcpy(m->key_tape_idx, key_idx, 2 * sizeof(int));
  m->beta_tape_idx = 6;
  m->out_tape_start = 8;

  /* dy = [1, 1] */
  double grad[10] = {0};
  grad[8] = 1.0;
  grad[9] = 1.0;
  tensor_batch_cossim_backward(grad, m);

  /* Check d_beta numerically */
  {
    arena_reset();
    BatchCosSimMeta *mp = batch_cossim_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    memcpy(mp->key_vals, key, 2 * sizeof(double));
    double out_p[2], out_m[2];
    mp->beta_val = beta + eps;
    batch_cossim_compute(mp, out_p);
    mp->beta_val = beta - eps;
    batch_cossim_compute(mp, out_m);
    double num = ((out_p[0] - out_m[0]) + (out_p[1] - out_m[1])) / (2 * eps);
    check_close("batch_cossim_bwd d_beta", grad[6], num, 1e-5);
  }

  /* Check d_key numerically */
  for (int j = 0; j < w; j++) {
    arena_reset();
    BatchCosSimMeta *mp = batch_cossim_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    mp->beta_val = beta;
    double out_p[2], out_m[2];
    double key_p[2], key_m[2];
    memcpy(key_p, key, 2 * sizeof(double));
    memcpy(key_m, key, 2 * sizeof(double));
    key_p[j] += eps; key_m[j] -= eps;
    memcpy(mp->key_vals, key_p, 2 * sizeof(double));
    batch_cossim_compute(mp, out_p);
    memcpy(mp->key_vals, key_m, 2 * sizeof(double));
    batch_cossim_compute(mp, out_m);
    double num = ((out_p[0] - out_m[0]) + (out_p[1] - out_m[1])) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "batch_cossim_bwd d_key[%d]", j);
    check_close(name, grad[4 + j], num, 1e-5);
  }

  /* Check d_mem numerically */
  for (int i = 0; i < n * w; i++) {
    arena_reset();
    BatchCosSimMeta *mp = batch_cossim_meta_alloc(n, w);
    memcpy(mp->key_vals, key, 2 * sizeof(double));
    mp->beta_val = beta;
    double out_p[2], out_m[2];
    double mem_p[4], mem_m[4];
    memcpy(mem_p, mem, 4 * sizeof(double));
    memcpy(mem_m, mem, 4 * sizeof(double));
    mem_p[i] += eps; mem_m[i] -= eps;
    memcpy(mp->mem_vals, mem_p, 4 * sizeof(double));
    batch_cossim_compute(mp, out_p);
    memcpy(mp->mem_vals, mem_m, 4 * sizeof(double));
    batch_cossim_compute(mp, out_m);
    double num = ((out_p[0] - out_m[0]) + (out_p[1] - out_m[1])) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "batch_cossim_bwd d_mem[%d]", i);
    check_close(name, grad[i], num, 1e-5);
  }
}


/* -------------------------------------------------------------------
   ReadOp tests
   ------------------------------------------------------------------- */

static void test_readop_forward(void) {
  /* mem = [[1, 2], [3, 4], [5, 6]], weights = [0.5, 0.3, 0.2]
   * out[0] = 0.5*1 + 0.3*3 + 0.2*5 = 2.4
   * out[1] = 0.5*2 + 0.3*4 + 0.2*6 = 3.4 */
  arena_reset();
  ReadOpMeta *m = readop_meta_alloc(3, 2);
  double mem[] = {1, 2, 3, 4, 5, 6};
  double w[] = {0.5, 0.3, 0.2};
  memcpy(m->mem_vals, mem, 6 * sizeof(double));
  memcpy(m->weight_vals, w, 3 * sizeof(double));

  double out[2] = {0};
  readop_compute(m, out);

  check_close("readop[0]", out[0], 2.4, 1e-10);
  check_close("readop[1]", out[1], 3.4, 1e-10);
}

static void test_readop_backward(void) {
  /* mem = [[1, 2], [3, 4]], weights = [0.6, 0.4]
   * out[0] = 0.6*1 + 0.4*3 = 1.8, out[1] = 0.6*2 + 0.4*4 = 2.8
   * dy = [1, 1]
   * d_weight[0] = 1*1 + 1*2 = 3, d_weight[1] = 1*3 + 1*4 = 7
   * d_mem[0][0] = 1*0.6 = 0.6, d_mem[0][1] = 1*0.6 = 0.6
   * d_mem[1][0] = 1*0.4 = 0.4, d_mem[1][1] = 1*0.4 = 0.4 */
  arena_reset();
  ReadOpMeta *m = readop_meta_alloc(2, 2);
  double mem[] = {1, 2, 3, 4};
  double w[] = {0.6, 0.4};
  memcpy(m->mem_vals, mem, 4 * sizeof(double));
  memcpy(m->weight_vals, w, 2 * sizeof(double));

  int mem_idx[] = {0, 1, 2, 3};
  int w_idx[] = {4, 5};
  memcpy(m->mem_tape_idx, mem_idx, 4 * sizeof(int));
  memcpy(m->weight_tape_idx, w_idx, 2 * sizeof(int));
  m->out_tape_start = 7;

  /* layout: 0-3=mem, 4-5=weights, 6=op, 7-8=output */
  double grad[9] = {0};
  grad[7] = 1.0;
  grad[8] = 1.0;

  tensor_readop_backward(grad, m);

  check_close("readop_bwd d_mem[0][0]", grad[0], 0.6, 1e-10);
  check_close("readop_bwd d_mem[0][1]", grad[1], 0.6, 1e-10);
  check_close("readop_bwd d_mem[1][0]", grad[2], 0.4, 1e-10);
  check_close("readop_bwd d_mem[1][1]", grad[3], 0.4, 1e-10);
  check_close("readop_bwd d_weight[0]", grad[4], 3.0, 1e-10);
  check_close("readop_bwd d_weight[1]", grad[5], 7.0, 1e-10);
}


/* -------------------------------------------------------------------
   WriteOp tests
   ------------------------------------------------------------------- */

static void test_writeop_forward(void) {
  /* mem = [[1, 1], [1, 1]], weights = [1, 0], erase = [1, 1], add = [0.5, 0.5]
   * out[0][0] = 1*(1-1*1) + 1*0.5 = 0.5
   * out[0][1] = 1*(1-1*1) + 1*0.5 = 0.5
   * out[1][0] = 1*(1-0*1) + 0*0.5 = 1.0
   * out[1][1] = 1*(1-0*1) + 0*0.5 = 1.0 */
  arena_reset();
  WriteOpMeta *m = writeop_meta_alloc(2, 2);
  double mem[] = {1, 1, 1, 1};
  double w[] = {1, 0};
  double e[] = {1, 1};
  double a[] = {0.5, 0.5};
  memcpy(m->mem_vals, mem, 4 * sizeof(double));
  memcpy(m->weight_vals, w, 2 * sizeof(double));
  memcpy(m->erase_vals, e, 2 * sizeof(double));
  memcpy(m->add_vals, a, 2 * sizeof(double));

  double out[4] = {0};
  writeop_compute(m, out);

  check_close("writeop[0][0]", out[0], 0.5, 1e-10);
  check_close("writeop[0][1]", out[1], 0.5, 1e-10);
  check_close("writeop[1][0]", out[2], 1.0, 1e-10);
  check_close("writeop[1][1]", out[3], 1.0, 1e-10);
}

static void test_writeop_backward(void) {
  /* Numerical gradient check for write operation.
   * mem = [[2, 3], [1, 4]], w = [0.7, 0.3], e = [0.5, 0.8], a = [1, -1] */
  arena_reset();
  double eps = 1e-5;
  int n = 2, w = 2;

  double mem[] = {2, 3, 1, 4};
  double wt[] = {0.7, 0.3};
  double er[] = {0.5, 0.8};
  double ad[] = {1, -1};

  /* Compute forward for analytical gradients */
  WriteOpMeta *m = writeop_meta_alloc(n, w);
  memcpy(m->mem_vals, mem, 4 * sizeof(double));
  memcpy(m->weight_vals, wt, 2 * sizeof(double));
  memcpy(m->erase_vals, er, 2 * sizeof(double));
  memcpy(m->add_vals, ad, 2 * sizeof(double));

  /* Tape: 0-3=mem, 4-5=weight, 6-7=erase, 8-9=add, 10=op, 11-14=output */
  int mem_idx[] = {0, 1, 2, 3};
  int wt_idx[] = {4, 5};
  int er_idx[] = {6, 7};
  int ad_idx[] = {8, 9};
  memcpy(m->mem_tape_idx, mem_idx, 4 * sizeof(int));
  memcpy(m->weight_tape_idx, wt_idx, 2 * sizeof(int));
  memcpy(m->erase_tape_idx, er_idx, 2 * sizeof(int));
  memcpy(m->add_tape_idx, ad_idx, 2 * sizeof(int));
  m->out_tape_start = 11;

  /* dy = [1, 1, 1, 1] */
  double grad[15] = {0};
  grad[11] = 1.0; grad[12] = 1.0; grad[13] = 1.0; grad[14] = 1.0;

  tensor_writeop_backward(grad, m);

  /* Numerical check for each input element. loss = sum(out) */
  /* Check d_mem */
  for (int i = 0; i < n * w; i++) {
    arena_reset();
    WriteOpMeta *mp = writeop_meta_alloc(n, w);
    memcpy(mp->weight_vals, wt, 2 * sizeof(double));
    memcpy(mp->erase_vals, er, 2 * sizeof(double));
    memcpy(mp->add_vals, ad, 2 * sizeof(double));
    double out_p[4], out_m[4], mem_p[4], mem_m[4];
    memcpy(mem_p, mem, 4 * sizeof(double));
    memcpy(mem_m, mem, 4 * sizeof(double));
    mem_p[i] += eps; mem_m[i] -= eps;
    memcpy(mp->mem_vals, mem_p, 4 * sizeof(double));
    writeop_compute(mp, out_p);
    memcpy(mp->mem_vals, mem_m, 4 * sizeof(double));
    writeop_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "writeop_bwd d_mem[%d]", i);
    check_close(name, grad[i], num, 1e-5);
  }

  /* Check d_weight */
  for (int i = 0; i < n; i++) {
    arena_reset();
    WriteOpMeta *mp = writeop_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    memcpy(mp->erase_vals, er, 2 * sizeof(double));
    memcpy(mp->add_vals, ad, 2 * sizeof(double));
    double out_p[4], out_m[4], wt_p[2], wt_m[2];
    memcpy(wt_p, wt, 2 * sizeof(double));
    memcpy(wt_m, wt, 2 * sizeof(double));
    wt_p[i] += eps; wt_m[i] -= eps;
    memcpy(mp->weight_vals, wt_p, 2 * sizeof(double));
    writeop_compute(mp, out_p);
    memcpy(mp->weight_vals, wt_m, 2 * sizeof(double));
    writeop_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "writeop_bwd d_weight[%d]", i);
    check_close(name, grad[4 + i], num, 1e-5);
  }

  /* Check d_erase */
  for (int j = 0; j < w; j++) {
    arena_reset();
    WriteOpMeta *mp = writeop_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    memcpy(mp->weight_vals, wt, 2 * sizeof(double));
    memcpy(mp->add_vals, ad, 2 * sizeof(double));
    double out_p[4], out_m[4], er_p[2], er_m[2];
    memcpy(er_p, er, 2 * sizeof(double));
    memcpy(er_m, er, 2 * sizeof(double));
    er_p[j] += eps; er_m[j] -= eps;
    memcpy(mp->erase_vals, er_p, 2 * sizeof(double));
    writeop_compute(mp, out_p);
    memcpy(mp->erase_vals, er_m, 2 * sizeof(double));
    writeop_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "writeop_bwd d_erase[%d]", j);
    check_close(name, grad[6 + j], num, 1e-5);
  }

  /* Check d_add */
  for (int j = 0; j < w; j++) {
    arena_reset();
    WriteOpMeta *mp = writeop_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    memcpy(mp->weight_vals, wt, 2 * sizeof(double));
    memcpy(mp->erase_vals, er, 2 * sizeof(double));
    double out_p[4], out_m[4], ad_p[2], ad_m[2];
    memcpy(ad_p, ad, 2 * sizeof(double));
    memcpy(ad_m, ad, 2 * sizeof(double));
    ad_p[j] += eps; ad_m[j] -= eps;
    memcpy(mp->add_vals, ad_p, 2 * sizeof(double));
    writeop_compute(mp, out_p);
    memcpy(mp->add_vals, ad_m, 2 * sizeof(double));
    writeop_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "writeop_bwd d_add[%d]", j);
    check_close(name, grad[8 + j], num, 1e-5);
  }
}


/* -------------------------------------------------------------------
   Interpolation write tests
   ------------------------------------------------------------------- */

static void test_interp_write_forward(void) {
  /* mem = [[1, 2], [3, 4]], weights = [0.8, 0.0], add = [5, 6]
   * raw[0][0] = (1-0.8)*1 + 0.8*5 = 4.2,  out = tanh(4.2)
   * raw[0][1] = (1-0.8)*2 + 0.8*6 = 5.2,  out = tanh(5.2)
   * raw[1][0] = (1-0.0)*3 + 0.0*5 = 3.0,  out = tanh(3.0)
   * raw[1][1] = (1-0.0)*4 + 0.0*6 = 4.0,  out = tanh(4.0) */
  arena_reset();
  InterpWriteMeta *m = interp_write_meta_alloc(2, 2);
  double mem[] = {1, 2, 3, 4};
  double w[] = {0.8, 0.0};
  double a[] = {5, 6};
  memcpy(m->mem_vals, mem, 4 * sizeof(double));
  memcpy(m->weight_vals, w, 2 * sizeof(double));
  memcpy(m->add_vals, a, 2 * sizeof(double));

  double out[4] = {0};
  interp_write_compute(m, out);

  check_close("interp_write[0][0]", out[0], tanh(4.2), 1e-10);
  check_close("interp_write[0][1]", out[1], tanh(5.2), 1e-10);
  check_close("interp_write[1][0]", out[2], tanh(3.0), 1e-10);
  check_close("interp_write[1][1]", out[3], tanh(4.0), 1e-10);
}

static void test_interp_write_backward(void) {
  /* Numerical gradient check for interpolation write.
   * mem = [[2, 3], [1, 4]], w = [0.7, 0.3], a = [1, -1] */
  arena_reset();
  double eps = 1e-5;
  int n = 2, w = 2;

  double mem[] = {2, 3, 1, 4};
  double wt[] = {0.7, 0.3};
  double ad[] = {1, -1};

  /* Compute forward for analytical gradients */
  InterpWriteMeta *m = interp_write_meta_alloc(n, w);
  memcpy(m->mem_vals, mem, 4 * sizeof(double));
  memcpy(m->weight_vals, wt, 2 * sizeof(double));
  memcpy(m->add_vals, ad, 2 * sizeof(double));

  /* Tape: 0-3=mem, 4-5=weight, 6-7=add, 8=op, 9-12=output */
  int mem_idx[] = {0, 1, 2, 3};
  int wt_idx[] = {4, 5};
  int ad_idx[] = {6, 7};
  memcpy(m->mem_tape_idx, mem_idx, 4 * sizeof(int));
  memcpy(m->weight_tape_idx, wt_idx, 2 * sizeof(int));
  memcpy(m->add_tape_idx, ad_idx, 2 * sizeof(int));
  m->out_tape_start = 9;

  /* Run forward to populate out_vals (needed for tanh derivative in backward) */
  double fwd_out[4];
  interp_write_compute(m, fwd_out);

  /* dy = [1, 1, 1, 1] */
  double grad[13] = {0};
  grad[9] = 1.0; grad[10] = 1.0; grad[11] = 1.0; grad[12] = 1.0;

  tensor_interp_write_backward(grad, m);

  /* Numerical check for d_mem */
  for (int i = 0; i < n * w; i++) {
    arena_reset();
    InterpWriteMeta *mp = interp_write_meta_alloc(n, w);
    memcpy(mp->weight_vals, wt, 2 * sizeof(double));
    memcpy(mp->add_vals, ad, 2 * sizeof(double));
    double out_p[4], out_m[4], mem_p[4], mem_m[4];
    memcpy(mem_p, mem, 4 * sizeof(double));
    memcpy(mem_m, mem, 4 * sizeof(double));
    mem_p[i] += eps; mem_m[i] -= eps;
    memcpy(mp->mem_vals, mem_p, 4 * sizeof(double));
    interp_write_compute(mp, out_p);
    memcpy(mp->mem_vals, mem_m, 4 * sizeof(double));
    interp_write_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "interp_write_bwd d_mem[%d]", i);
    check_close(name, grad[i], num, 1e-5);
  }

  /* Check d_weight */
  for (int i = 0; i < n; i++) {
    arena_reset();
    InterpWriteMeta *mp = interp_write_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    memcpy(mp->add_vals, ad, 2 * sizeof(double));
    double out_p[4], out_m[4], wt_p[2], wt_m[2];
    memcpy(wt_p, wt, 2 * sizeof(double));
    memcpy(wt_m, wt, 2 * sizeof(double));
    wt_p[i] += eps; wt_m[i] -= eps;
    memcpy(mp->weight_vals, wt_p, 2 * sizeof(double));
    interp_write_compute(mp, out_p);
    memcpy(mp->weight_vals, wt_m, 2 * sizeof(double));
    interp_write_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "interp_write_bwd d_weight[%d]", i);
    check_close(name, grad[4 + i], num, 1e-5);
  }

  /* Check d_add */
  for (int j = 0; j < w; j++) {
    arena_reset();
    InterpWriteMeta *mp = interp_write_meta_alloc(n, w);
    memcpy(mp->mem_vals, mem, 4 * sizeof(double));
    memcpy(mp->weight_vals, wt, 2 * sizeof(double));
    double out_p[4], out_m[4], ad_p[2], ad_m[2];
    memcpy(ad_p, ad, 2 * sizeof(double));
    memcpy(ad_m, ad, 2 * sizeof(double));
    ad_p[j] += eps; ad_m[j] -= eps;
    memcpy(mp->add_vals, ad_p, 2 * sizeof(double));
    interp_write_compute(mp, out_p);
    memcpy(mp->add_vals, ad_m, 2 * sizeof(double));
    interp_write_compute(mp, out_m);
    double num = 0;
    for (int k = 0; k < n * w; k++) num += (out_p[k] - out_m[k]) / (2 * eps);
    char name[64];
    snprintf(name, sizeof(name), "interp_write_bwd d_add[%d]", j);
    check_close(name, grad[6 + j], num, 1e-5);
  }
}


/* -------------------------------------------------------------------
   NtmMemBuf tests
   ------------------------------------------------------------------- */

static void test_ntm_mem_alloc(void) {
  NtmMemBuf *mb = ntm_mem_alloc(3, 2);
  check("ntm_mem_alloc_non_null", mb != NULL);
  check("ntm_mem_n", ntm_mem_get_n(mb) == 3);
  check("ntm_mem_w", ntm_mem_get_w(mb) == 2);
  check("ntm_mem_cached_gen_init", ntm_mem_cached_gen(mb) == -1);

  /* Values should be zero-initialized */
  double *vals = ntm_mem_vals_ptr(mb);
  check_close("ntm_mem_val_init_0", vals[0], 0.0, 1e-12);
  check_close("ntm_mem_val_init_5", vals[5], 0.0, 1e-12);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_set_val(void) {
  NtmMemBuf *mb = ntm_mem_alloc(2, 3);
  ntm_mem_set_val(mb, 0, 1.5);
  ntm_mem_set_val(mb, 3, 2.7);
  ntm_mem_set_val(mb, 5, -0.3);

  double *vals = ntm_mem_vals_ptr(mb);
  check_close("ntm_mem_set_val[0]", vals[0], 1.5, 1e-12);
  check_close("ntm_mem_set_val[3]", vals[3], 2.7, 1e-12);
  check_close("ntm_mem_set_val[5]", vals[5], -0.3, 1e-12);
  /* Unchanged elements stay zero */
  check_close("ntm_mem_set_val[1]", vals[1], 0.0, 1e-12);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_set_pid(void) {
  NtmMemBuf *mb = ntm_mem_alloc(2, 2);
  ntm_mem_set_pid(mb, 0, "mem0");
  ntm_mem_set_pid(mb, 1, "mem1");
  ntm_mem_set_pid(mb, 2, "mem2");
  ntm_mem_set_pid(mb, 3, "mem3");

  char **pids = ntm_mem_pids_ptr(mb);
  check("ntm_mem_pid_0", strcmp(pids[0], "mem0") == 0);
  check("ntm_mem_pid_3", strcmp(pids[3], "mem3") == 0);
  /* Same string should be interned (same pointer) */
  ntm_mem_set_pid(mb, 0, "mem0");
  check("ntm_mem_pid_intern", pids[0] == ntm_mem_pids_ptr(mb)[0]);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_update_vals(void) {
  NtmMemBuf *mb = ntm_mem_alloc(2, 3);
  double new_vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  ntm_mem_update_vals(mb, new_vals);

  double *vals = ntm_mem_vals_ptr(mb);
  check_close("ntm_mem_update_vals[0]", vals[0], 1.0, 1e-12);
  check_close("ntm_mem_update_vals[2]", vals[2], 3.0, 1e-12);
  check_close("ntm_mem_update_vals[5]", vals[5], 6.0, 1e-12);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_update_tape_idx(void) {
  NtmMemBuf *mb = ntm_mem_alloc(2, 3);
  ntm_mem_update_tape_idx(mb, 100);

  int *idx = ntm_mem_tape_idx_ptr(mb);
  check("ntm_mem_tape_idx[0]", idx[0] == 100);
  check("ntm_mem_tape_idx[3]", idx[3] == 103);
  check("ntm_mem_tape_idx[5]", idx[5] == 105);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_sync_vals(void) {
  NtmMemBuf *mb = ntm_mem_alloc(2, 2);
  ntm_mem_set_cached(mb, 5, 42);
  check("ntm_mem_cached_before_sync", ntm_mem_cached_gen(mb) == 5);

  double new_vals[] = {1.1, 2.2, 3.3, 4.4};
  ntm_mem_sync_vals(mb, new_vals, 4);

  double *vals = ntm_mem_vals_ptr(mb);
  check_close("ntm_mem_sync[0]", vals[0], 1.1, 1e-12);
  check_close("ntm_mem_sync[3]", vals[3], 4.4, 1e-12);
  /* Sync should invalidate cache */
  check("ntm_mem_cached_after_sync", ntm_mem_cached_gen(mb) == -1);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_pack_batch_cossim(void) {
  /* Set up NtmMemBuf with known values and tape indices */
  NtmMemBuf *mb = ntm_mem_alloc(2, 3);
  double mem_vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  ntm_mem_update_vals(mb, mem_vals);
  ntm_mem_update_tape_idx(mb, 10);

  /* Allocate meta and pack from buffer */
  arena_reset();
  BatchCosSimMeta *meta = batch_cossim_meta_alloc(2, 3);
  batch_cossim_pack_mem_buf(meta, mb);

  /* Verify values were copied */
  check_close("pack_bcs_val[0]", meta->mem_vals[0], 1.0, 1e-12);
  check_close("pack_bcs_val[5]", meta->mem_vals[5], 6.0, 1e-12);
  check("pack_bcs_idx[0]", meta->mem_tape_idx[0] == 10);
  check("pack_bcs_idx[5]", meta->mem_tape_idx[5] == 15);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_pack_readop(void) {
  NtmMemBuf *mb = ntm_mem_alloc(3, 2);
  double mem_vals[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  ntm_mem_update_vals(mb, mem_vals);
  ntm_mem_update_tape_idx(mb, 20);

  arena_reset();
  ReadOpMeta *meta = readop_meta_alloc(3, 2);
  readop_pack_mem_buf(meta, mb);

  check_close("pack_ro_val[0]", meta->mem_vals[0], 0.1, 1e-12);
  check_close("pack_ro_val[5]", meta->mem_vals[5], 0.6, 1e-12);
  check("pack_ro_idx[0]", meta->mem_tape_idx[0] == 20);
  check("pack_ro_idx[5]", meta->mem_tape_idx[5] == 25);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_pack_interp_write(void) {
  NtmMemBuf *mb = ntm_mem_alloc(2, 2);
  double mem_vals[] = {10.0, 20.0, 30.0, 40.0};
  ntm_mem_update_vals(mb, mem_vals);
  ntm_mem_update_tape_idx(mb, 50);

  arena_reset();
  InterpWriteMeta *meta = interp_write_meta_alloc(2, 2);
  interp_write_pack_mem_buf(meta, mb);

  check_close("pack_iw_val[0]", meta->mem_vals[0], 10.0, 1e-12);
  check_close("pack_iw_val[3]", meta->mem_vals[3], 40.0, 1e-12);
  check("pack_iw_idx[0]", meta->mem_tape_idx[0] == 50);
  check("pack_iw_idx[3]", meta->mem_tape_idx[3] == 53);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
}

static void test_ntm_mem_full_lifecycle(void) {
  /* Simulate: alloc -> init -> ensure (fake) -> pack -> interp_write update -> pack again */
  NtmMemBuf *mb = ntm_mem_alloc(2, 2);

  /* 1. Init with values (like nameParams) */
  ntm_mem_set_val(mb, 0, 1e-6);
  ntm_mem_set_val(mb, 1, 1e-6);
  ntm_mem_set_val(mb, 2, 1e-6);
  ntm_mem_set_val(mb, 3, 1e-6);
  ntm_mem_set_pid(mb, 0, "ntm0_mem0");
  ntm_mem_set_pid(mb, 1, "ntm0_mem1");
  ntm_mem_set_pid(mb, 2, "ntm0_mem2");
  ntm_mem_set_pid(mb, 3, "ntm0_mem3");

  /* 2. Simulate ensure_on_tape: tape indices 0-3 */
  ntm_mem_update_tape_idx(mb, 0);
  ntm_mem_set_cached(mb, 0, 0);

  /* 3. Pack into BatchCosSim meta */
  arena_reset();
  BatchCosSimMeta *bcs = batch_cossim_meta_alloc(2, 2);
  batch_cossim_pack_mem_buf(bcs, mb);
  check_close("lifecycle_bcs_val", bcs->mem_vals[0], 1e-6, 1e-12);
  check("lifecycle_bcs_idx", bcs->mem_tape_idx[0] == 0);

  /* 4. Simulate InterpWrite output: new memory values */
  double new_vals[] = {0.5, -0.3, 0.8, 0.1};
  ntm_mem_update_vals(mb, new_vals);
  ntm_mem_update_tape_idx(mb, 200); /* new ConstOps at positions 200-203 */

  /* 5. Pack into ReadOp meta (should see updated values) */
  arena_reset();
  ReadOpMeta *ro = readop_meta_alloc(2, 2);
  readop_pack_mem_buf(ro, mb);
  check_close("lifecycle_ro_val[0]", ro->mem_vals[0], 0.5, 1e-12);
  check_close("lifecycle_ro_val[1]", ro->mem_vals[1], -0.3, 1e-12);
  check("lifecycle_ro_idx[0]", ro->mem_tape_idx[0] == 200);
  check("lifecycle_ro_idx[3]", ro->mem_tape_idx[3] == 203);

  /* 6. Sync after applyDeltas (new epoch) */
  double updated[] = {0.51, -0.29, 0.79, 0.11};
  ntm_mem_sync_vals(mb, updated, 4);
  check_close("lifecycle_sync_val", ntm_mem_vals_ptr(mb)[0], 0.51, 1e-12);
  check("lifecycle_sync_invalidated", ntm_mem_cached_gen(mb) == -1);

  free(mb->vals); free(mb->tape_idx); free(mb->pids); free(mb);
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
   InterpolateOp tests
   ------------------------------------------------------------------- */

static void test_interpolate_forward(void) {
  arena_reset();
  /* n=3, g=0.4, content=[1,2,3], prev=[4,5,6] */
  InterpolateMeta *m = interpolate_meta_alloc(3);
  interpolate_meta_set_g(m, 0.4, 0);
  m->content_vals[0] = 1.0; m->content_vals[1] = 2.0; m->content_vals[2] = 3.0;
  m->prev_vals[0] = 4.0; m->prev_vals[1] = 5.0; m->prev_vals[2] = 6.0;
  double out[3];
  interpolate_compute(m, out);
  /* out[i] = 0.4*content[i] + 0.6*prev[i] */
  check_close("interpolate[0]", out[0], 0.4*1.0 + 0.6*4.0, 1e-10);
  check_close("interpolate[1]", out[1], 0.4*2.0 + 0.6*5.0, 1e-10);
  check_close("interpolate[2]", out[2], 0.4*3.0 + 0.6*6.0, 1e-10);
}

static void test_interpolate_backward(void) {
  arena_reset();
  tape_init();
  tape_gen = 1;
  tape_size = 0;
  /* 7 tape entries: g=idx0, content=[idx1,2,3], prev=[idx4,5,6], out=[idx7,8,9] */
  int n = 3;
  InterpolateMeta *m = interpolate_meta_alloc(n);
  double g = 0.4;
  interpolate_meta_set_g(m, g, 0);
  m->content_vals[0] = 1.0; m->content_vals[1] = 2.0; m->content_vals[2] = 3.0;
  m->content_tape_idx[0] = 1; m->content_tape_idx[1] = 2; m->content_tape_idx[2] = 3;
  m->prev_vals[0] = 4.0; m->prev_vals[1] = 5.0; m->prev_vals[2] = 6.0;
  m->prev_tape_idx[0] = 4; m->prev_tape_idx[1] = 5; m->prev_tape_idx[2] = 6;
  m->out_tape_start = 7;
  double grad[10] = {0};
  grad[7] = 1.0; grad[8] = 0.5; grad[9] = 0.25;
  tensor_interpolate_backward(grad, m);
  /* d_g = sum(dy[i] * (content[i] - prev[i])) */
  double expected_dg = 1.0*(1-4) + 0.5*(2-5) + 0.25*(3-6);
  check_close("interpolate_bwd d_g", grad[0], expected_dg, 1e-10);
  /* d_content[i] = dy[i] * g */
  check_close("interpolate_bwd d_content[0]", grad[1], 1.0*0.4, 1e-10);
  check_close("interpolate_bwd d_content[1]", grad[2], 0.5*0.4, 1e-10);
  check_close("interpolate_bwd d_content[2]", grad[3], 0.25*0.4, 1e-10);
  /* d_prev[i] = dy[i] * (1-g) */
  check_close("interpolate_bwd d_prev[0]", grad[4], 1.0*0.6, 1e-10);
  check_close("interpolate_bwd d_prev[1]", grad[5], 0.5*0.6, 1e-10);
  check_close("interpolate_bwd d_prev[2]", grad[6], 0.25*0.6, 1e-10);
}


/* -------------------------------------------------------------------
   ShiftOp tests
   ------------------------------------------------------------------- */

static void test_shift_forward(void) {
  arena_reset();
  /* n=4, kernel=[0.1, 0.7, 0.2] (sl, ss, sr), input=[1,2,3,4] */
  ShiftMeta *m = shift_meta_alloc(4);
  m->kernel_vals[0] = 0.1; m->kernel_vals[1] = 0.7; m->kernel_vals[2] = 0.2;
  m->input_vals[0] = 1.0; m->input_vals[1] = 2.0; m->input_vals[2] = 3.0; m->input_vals[3] = 4.0;
  double out[4];
  shift_compute(m, out);
  /* out[i] = sl*input[(i+1)%n] + ss*input[i] + sr*input[(i+n-1)%n] */
  check_close("shift[0]", out[0], 0.1*2.0 + 0.7*1.0 + 0.2*4.0, 1e-10);
  check_close("shift[1]", out[1], 0.1*3.0 + 0.7*2.0 + 0.2*1.0, 1e-10);
  check_close("shift[2]", out[2], 0.1*4.0 + 0.7*3.0 + 0.2*2.0, 1e-10);
  check_close("shift[3]", out[3], 0.1*1.0 + 0.7*4.0 + 0.2*3.0, 1e-10);
}

static void test_shift_backward(void) {
  arena_reset();
  int n = 4;
  ShiftMeta *m = shift_meta_alloc(n);
  m->kernel_vals[0] = 0.1; m->kernel_vals[1] = 0.7; m->kernel_vals[2] = 0.2;
  m->kernel_tape_idx[0] = 0; m->kernel_tape_idx[1] = 1; m->kernel_tape_idx[2] = 2;
  m->input_vals[0] = 1.0; m->input_vals[1] = 2.0; m->input_vals[2] = 3.0; m->input_vals[3] = 4.0;
  m->input_tape_idx[0] = 3; m->input_tape_idx[1] = 4; m->input_tape_idx[2] = 5; m->input_tape_idx[3] = 6;
  m->out_tape_start = 7;
  double grad[11] = {0};
  grad[7] = 1.0; grad[8] = 1.0; grad[9] = 1.0; grad[10] = 1.0;
  tensor_shift_backward(grad, m);
  /* dk0 = sum(dy[i] * input[(i+1)%n]) = 1*(2)+1*(3)+1*(4)+1*(1) = 10 */
  check_close("shift_bwd dk0", grad[0], 10.0, 1e-10);
  /* dk1 = sum(dy[i] * input[i]) = 1+2+3+4 = 10 */
  check_close("shift_bwd dk1", grad[1], 10.0, 1e-10);
  /* dk2 = sum(dy[i] * input[(i+n-1)%n]) = 1*(4)+1*(1)+1*(2)+1*(3) = 10 */
  check_close("shift_bwd dk2", grad[2], 10.0, 1e-10);
  /* d_input[j]: accumulated from all outputs that reference it
     input[0] used by: out[3] as fwd (sl=0.1), out[0] as ss (0.7), out[1] as bwd (0.2)
     = 1*0.1 + 1*0.7 + 1*0.2 = 1.0 */
  check_close("shift_bwd d_input[0]", grad[3], 1.0, 1e-10);
  check_close("shift_bwd d_input[1]", grad[4], 1.0, 1e-10);
  check_close("shift_bwd d_input[2]", grad[5], 1.0, 1e-10);
  check_close("shift_bwd d_input[3]", grad[6], 1.0, 1e-10);
}


/* -------------------------------------------------------------------
   FocusOp tests
   ------------------------------------------------------------------- */

static void test_focus_forward(void) {
  arena_reset();
  /* n=3, gamma=2.0, input=[1,2,3] */
  FocusMeta *m = focus_meta_alloc(3);
  focus_meta_set_gamma(m, 2.0, 0);
  m->input_vals[0] = 1.0; m->input_vals[1] = 2.0; m->input_vals[2] = 3.0;
  double out[3];
  focus_compute(m, out);
  /* raised = [1^2, 2^2, 3^2] = [1,4,9], sum=14 */
  check_close("focus[0]", out[0], 1.0/14.0, 1e-10);
  check_close("focus[1]", out[1], 4.0/14.0, 1e-10);
  check_close("focus[2]", out[2], 9.0/14.0, 1e-10);
}

static void test_focus_backward(void) {
  arena_reset();
  int n = 3;
  FocusMeta *m = focus_meta_alloc(n);
  double gamma = 2.0;
  focus_meta_set_gamma(m, gamma, 0);
  m->input_vals[0] = 1.0; m->input_vals[1] = 2.0; m->input_vals[2] = 3.0;
  m->input_tape_idx[0] = 1; m->input_tape_idx[1] = 2; m->input_tape_idx[2] = 3;
  m->out_tape_start = 4;
  /* Run forward first to populate raised_vals, sum_raised */
  double fwd_out[3];
  focus_compute(m, fwd_out);

  double grad[7] = {0};
  /* dy = [1, 0, 0] to test gradient for just the first output */
  grad[4] = 1.0;
  tensor_focus_backward(grad, m);
  /* Numerical gradient check: perturb input[0] by eps */
  double eps = 1e-6;
  m->input_vals[0] = 1.0 + eps;
  double out_plus[3];
  focus_compute(m, out_plus);
  m->input_vals[0] = 1.0 - eps;
  double out_minus[3];
  focus_compute(m, out_minus);
  m->input_vals[0] = 1.0;
  double num_grad = (out_plus[0] - out_minus[0]) / (2.0 * eps);
  check_close("focus_bwd d_input[0]", grad[1], num_grad, 1e-5);

  /* Also verify d_gamma with numerical gradient */
  double grad2[7] = {0};
  grad2[4] = 1.0;
  focus_meta_set_gamma(m, gamma, 0);
  focus_compute(m, fwd_out);
  tensor_focus_backward(grad2, m);

  focus_meta_set_gamma(m, gamma + eps, 0);
  focus_compute(m, out_plus);
  focus_meta_set_gamma(m, gamma - eps, 0);
  focus_compute(m, out_minus);
  double num_dgamma = (out_plus[0] - out_minus[0]) / (2.0 * eps);
  check_close("focus_bwd d_gamma", grad2[0], num_dgamma, 1e-5);
}


/* -------------------------------------------------------------------
   LstmCellOp tests
   ------------------------------------------------------------------- */

static void test_lstm_cell_forward(void) {
  arena_reset();
  /* o=2: 4*o=8 combined, prev_cell[2], output: newCell[2]+newHidden[2] */
  int o = 2;
  LstmCellMeta *m = lstm_cell_meta_alloc(o);

  /* mulIW = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] */
  /* mulRW = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75] */
  /* bias  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08] */
  for (int k = 0; k < 8; k++) {
    m->muliw_vals[k] = 0.1 * (k + 1);
    m->mulrw_vals[k] = 0.05 + 0.1 * k;
    m->bias_vals[k] = 0.01 * (k + 1);
  }
  m->prev_cell_vals[0] = 0.5;
  m->prev_cell_vals[1] = -0.3;

  double out[4];
  lstm_cell_compute(m, out);

  /* Verify manually:
   * combined[k] = mulIW[k] + mulRW[k] + bias[k]
   * combined = [0.16, 0.37, 0.58, 0.79, 1.0, 1.21, 1.42, 1.63] */
  double comb[8];
  for (int k = 0; k < 8; k++)
    comb[k] = m->muliw_vals[k] + m->mulrw_vals[k] + m->bias_vals[k];

  /* iGate = sigmoid(combined[0..2)) */
  double iG0 = 1.0 / (1.0 + exp(-comb[0]));
  double iG1 = 1.0 / (1.0 + exp(-comb[1]));
  /* fGate = sigmoid(combined[2..4)) */
  double fG0 = 1.0 / (1.0 + exp(-comb[2]));
  double fG1 = 1.0 / (1.0 + exp(-comb[3]));
  /* gGate = tanh(combined[4..6)) */
  double gG0 = tanh(comb[4]);
  double gG1 = tanh(comb[5]);
  /* oGate = sigmoid(combined[6..8)) */
  double oG0 = 1.0 / (1.0 + exp(-comb[6]));
  double oG1 = 1.0 / (1.0 + exp(-comb[7]));

  double nc0 = fG0 * 0.5 + iG0 * gG0;
  double nc1 = fG1 * (-0.3) + iG1 * gG1;
  double nh0 = oG0 * tanh(nc0);
  double nh1 = oG1 * tanh(nc1);

  check_close("lstm_cell newCell[0]", out[0], nc0, 1e-12);
  check_close("lstm_cell newCell[1]", out[1], nc1, 1e-12);
  check_close("lstm_cell newHidden[0]", out[2], nh0, 1e-12);
  check_close("lstm_cell newHidden[1]", out[3], nh1, 1e-12);
}

static void test_lstm_cell_backward(void) {
  /* Numerical gradient check for LstmCellOp.
   * Use o=2 for simplicity. Check gradients for all inputs:
   * mulIW[8], mulRW[8], bias[8], prevCell[2]
   * Outputs: newCell[2] + newHidden[2] at tape indices offset by +1 */
  arena_reset();
  int o = 2;
  int fo = 8; /* 4*o */

  /* Base values */
  double muliw[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  double mulrw[8] = {0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75};
  double bias[8]  = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08};
  double pcell[2] = {0.5, -0.3};

  /* Tape layout: muliw[0..7] = idx 0..7, mulrw[8..15] = idx 8..15,
   * bias[16..23] = idx 16..23, prevCell[24..25] = idx 24..25,
   * op entry = idx 26, outputs = idx 27..30 */
  int n_inputs = 26; /* 8+8+8+2 */

  /* Analytical backward */
  LstmCellMeta *m = lstm_cell_meta_alloc(o);
  for (int k = 0; k < fo; k++) {
    m->muliw_vals[k] = muliw[k];
    m->mulrw_vals[k] = mulrw[k];
    m->bias_vals[k] = bias[k];
    m->muliw_tape_idx[k] = k;
    m->mulrw_tape_idx[k] = fo + k;
    m->bias_tape_idx[k] = 2 * fo + k;
  }
  for (int j = 0; j < o; j++) {
    m->prev_cell_vals[j] = pcell[j];
    m->prev_cell_tape_idx[j] = 3 * fo + j;
  }
  m->out_tape_start = n_inputs + 1; /* op entry at idx 26, outputs at 27..30 */

  double out[4];
  lstm_cell_compute(m, out);

  /* Seed gradient: 1.0 for all outputs (at indices 27,28,29,30) */
  int total = n_inputs + 1 + 2 * o; /* 26 inputs + 1 op entry + 4 outputs */
  double *grad = (double *)calloc(total, sizeof(double));
  for (int k = 0; k < 2 * o; k++)
    grad[n_inputs + 1 + k] = 1.0; /* outputs start at op+1 */

  tensor_lstm_cell_backward(grad, m);

  /* Numerical gradient check */
  double eps = 1e-5;
  double out_plus[4], out_minus[4];

  /* Check mulIW gradients */
  for (int k = 0; k < fo; k++) {
    LstmCellMeta *m2 = lstm_cell_meta_alloc(o);
    for (int j = 0; j < fo; j++) {
      m2->muliw_vals[j] = muliw[j];
      m2->mulrw_vals[j] = mulrw[j];
      m2->bias_vals[j] = bias[j];
    }
    for (int j = 0; j < o; j++) m2->prev_cell_vals[j] = pcell[j];

    m2->muliw_vals[k] = muliw[k] + eps;
    lstm_cell_compute(m2, out_plus);
    m2->muliw_vals[k] = muliw[k] - eps;
    lstm_cell_compute(m2, out_minus);

    double num = 0.0;
    for (int j = 0; j < 2 * o; j++)
      num += (out_plus[j] - out_minus[j]) / (2.0 * eps);

    char name[64];
    snprintf(name, sizeof(name), "lstm_cell_bwd d_muliw[%d]", k);
    check_close(name, grad[k], num, 1e-5);
  }

  /* Check mulRW gradients */
  for (int k = 0; k < fo; k++) {
    LstmCellMeta *m2 = lstm_cell_meta_alloc(o);
    for (int j = 0; j < fo; j++) {
      m2->muliw_vals[j] = muliw[j];
      m2->mulrw_vals[j] = mulrw[j];
      m2->bias_vals[j] = bias[j];
    }
    for (int j = 0; j < o; j++) m2->prev_cell_vals[j] = pcell[j];

    m2->mulrw_vals[k] = mulrw[k] + eps;
    lstm_cell_compute(m2, out_plus);
    m2->mulrw_vals[k] = mulrw[k] - eps;
    lstm_cell_compute(m2, out_minus);

    double num = 0.0;
    for (int j = 0; j < 2 * o; j++)
      num += (out_plus[j] - out_minus[j]) / (2.0 * eps);

    char name[64];
    snprintf(name, sizeof(name), "lstm_cell_bwd d_mulrw[%d]", k);
    check_close(name, grad[fo + k], num, 1e-5);
  }

  /* Check bias gradients */
  for (int k = 0; k < fo; k++) {
    LstmCellMeta *m2 = lstm_cell_meta_alloc(o);
    for (int j = 0; j < fo; j++) {
      m2->muliw_vals[j] = muliw[j];
      m2->mulrw_vals[j] = mulrw[j];
      m2->bias_vals[j] = bias[j];
    }
    for (int j = 0; j < o; j++) m2->prev_cell_vals[j] = pcell[j];

    m2->bias_vals[k] = bias[k] + eps;
    lstm_cell_compute(m2, out_plus);
    m2->bias_vals[k] = bias[k] - eps;
    lstm_cell_compute(m2, out_minus);

    double num = 0.0;
    for (int j = 0; j < 2 * o; j++)
      num += (out_plus[j] - out_minus[j]) / (2.0 * eps);

    char name[64];
    snprintf(name, sizeof(name), "lstm_cell_bwd d_bias[%d]", k);
    check_close(name, grad[2 * fo + k], num, 1e-5);
  }

  /* Check prevCell gradients */
  for (int k = 0; k < o; k++) {
    LstmCellMeta *m2 = lstm_cell_meta_alloc(o);
    for (int j = 0; j < fo; j++) {
      m2->muliw_vals[j] = muliw[j];
      m2->mulrw_vals[j] = mulrw[j];
      m2->bias_vals[j] = bias[j];
    }
    for (int j = 0; j < o; j++) m2->prev_cell_vals[j] = pcell[j];

    m2->prev_cell_vals[k] = pcell[k] + eps;
    lstm_cell_compute(m2, out_plus);
    m2->prev_cell_vals[k] = pcell[k] - eps;
    lstm_cell_compute(m2, out_minus);

    double num = 0.0;
    for (int j = 0; j < 2 * o; j++)
      num += (out_plus[j] - out_minus[j]) / (2.0 * eps);

    char name[64];
    snprintf(name, sizeof(name), "lstm_cell_bwd d_prevcell[%d]", k);
    check_close(name, grad[3 * fo + k], num, 1e-5);
  }

  free(grad);
}


/* -------------------------------------------------------------------
   MatVec+Bias tests
   ------------------------------------------------------------------- */

static void test_matvec_bias_forward(void) {
  /* W = [[1, 2], [3, 4]], x = [5, 6], bias = [10, 20]
   * out = W*x + bias = [17+10, 39+20] = [27, 59] */
  arena_reset();
  double w[] = {1, 2, 3, 4};
  double x[] = {5, 6};
  double bias[] = {10, 20};

  MatVecMeta *meta = matvec_meta_alloc_buf_bias(2, 2, w, 0, bias, 4);
  memcpy(meta->x_vals, x, 2 * sizeof(double));
  int x_idx[] = {6, 7};
  memcpy(meta->x_tape_idx, x_idx, 2 * sizeof(int));

  double out[2];
  matvec_meta_compute(meta, out);

  check_close("matvec_bias_fwd out[0]", out[0], 27.0, 1e-12);
  check_close("matvec_bias_fwd out[1]", out[1], 59.0, 1e-12);
}

static void test_matvec_bias_backward(void) {
  /* W = [[1, 2], [3, 4]], x = [5, 6], bias = [10, 20]
   * out = [27, 59], dy = [1, 1]
   * dW = [[5, 6], [5, 6]], dx = [4, 6], dbias = [1, 1] */
  arena_reset();
  double w[] = {1, 2, 3, 4};
  double x[] = {5, 6};
  double bias[] = {10, 20};

  MatVecMeta *meta = matvec_meta_alloc_buf_bias(2, 2, w, 0, bias, 4);
  memcpy(meta->x_vals, x, 2 * sizeof(double));
  int x_idx[] = {6, 7};
  memcpy(meta->x_tape_idx, x_idx, 2 * sizeof(int));
  meta->out_tape_start = 9;

  /* grad: 0-3=W, 4-5=bias, 6-7=x, 8=op, 9-10=output */
  double grad[11] = {0};
  grad[9] = 1.0;  /* dy[0] */
  grad[10] = 1.0; /* dy[1] */

  tensor_matvec_backward(grad, meta);

  check_close("matvec_bias_bwd dW[0][0]", grad[0], 5.0, 1e-12);
  check_close("matvec_bias_bwd dW[0][1]", grad[1], 6.0, 1e-12);
  check_close("matvec_bias_bwd dW[1][0]", grad[2], 5.0, 1e-12);
  check_close("matvec_bias_bwd dW[1][1]", grad[3], 6.0, 1e-12);
  check_close("matvec_bias_bwd dbias[0]", grad[4], 1.0, 1e-12);
  check_close("matvec_bias_bwd dbias[1]", grad[5], 1.0, 1e-12);
  check_close("matvec_bias_bwd dx[0]", grad[6], 4.0, 1e-12);
  check_close("matvec_bias_bwd dx[1]", grad[7], 6.0, 1e-12);
}

/* -------------------------------------------------------------------
   LstmCell with bias buffer tests
   ------------------------------------------------------------------- */

static void test_lstm_cell_bias_buf_forward(void) {
  /* Same as test_lstm_cell_forward but using bias buffer path */
  arena_reset();
  int o = 2;
  double bias[8];
  for (int k = 0; k < 8; k++) bias[k] = 0.01 * (k + 1);

  LstmCellMeta *m = lstm_cell_meta_alloc_buf(o, bias, 100);

  for (int k = 0; k < 8; k++) {
    m->muliw_vals[k] = 0.1 * (k + 1);
    m->mulrw_vals[k] = 0.05 + 0.1 * k;
  }
  m->prev_cell_vals[0] = 0.5;
  m->prev_cell_vals[1] = -0.3;

  double out[4];
  lstm_cell_compute(m, out);

  /* Verify: same as non-buf path */
  double comb[8];
  for (int k = 0; k < 8; k++)
    comb[k] = m->muliw_vals[k] + m->mulrw_vals[k] + bias[k];

  double iG0 = 1.0 / (1.0 + exp(-comb[0]));
  double iG1 = 1.0 / (1.0 + exp(-comb[1]));
  double fG0 = 1.0 / (1.0 + exp(-comb[2]));
  double fG1 = 1.0 / (1.0 + exp(-comb[3]));
  double gG0 = tanh(comb[4]);
  double gG1 = tanh(comb[5]);
  double oG0 = 1.0 / (1.0 + exp(-comb[6]));
  double oG1 = 1.0 / (1.0 + exp(-comb[7]));

  double nc0 = fG0 * 0.5 + iG0 * gG0;
  double nc1 = fG1 * (-0.3) + iG1 * gG1;
  double nh0 = oG0 * tanh(nc0);
  double nh1 = oG1 * tanh(nc1);

  check_close("lstm_cell_buf newCell[0]", out[0], nc0, 1e-12);
  check_close("lstm_cell_buf newCell[1]", out[1], nc1, 1e-12);
  check_close("lstm_cell_buf newHidden[0]", out[2], nh0, 1e-12);
  check_close("lstm_cell_buf newHidden[1]", out[3], nh1, 1e-12);
}

static void test_lstm_cell_bias_buf_backward(void) {
  /* Numerical gradient check for LstmCellOp with bias buffer.
   * Same as test_lstm_cell_backward but using use_bias_buf=1. */
  arena_reset();
  int o = 2;
  int fo = 8;

  double muliw[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  double mulrw[8] = {0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75};
  double bias[8]  = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08};
  double pcell[2] = {0.5, -0.3};

  /* Tape layout: muliw[0..7], mulrw[8..15], bias[16..23], prevCell[24..25],
   * op=26, outputs=27..30 */
  int n_inputs = 26;
  int bias_tape_start = 16;

  LstmCellMeta *m = lstm_cell_meta_alloc_buf(o, bias, bias_tape_start);
  for (int k = 0; k < fo; k++) {
    m->muliw_vals[k] = muliw[k];
    m->mulrw_vals[k] = mulrw[k];
    m->muliw_tape_idx[k] = k;
    m->mulrw_tape_idx[k] = fo + k;
  }
  for (int j = 0; j < o; j++) {
    m->prev_cell_vals[j] = pcell[j];
    m->prev_cell_tape_idx[j] = 3 * fo + j;
  }
  m->out_tape_start = n_inputs + 1;

  double out[4];
  lstm_cell_compute(m, out);

  int total = n_inputs + 1 + 2 * o;
  double *grad = (double *)calloc(total, sizeof(double));
  for (int k = 0; k < 2 * o; k++)
    grad[n_inputs + 1 + k] = 1.0;

  tensor_lstm_cell_backward(grad, m);

  /* Numerical gradient check for bias */
  double eps = 1e-5;
  double out_plus[4], out_minus[4];

  for (int k = 0; k < fo; k++) {
    LstmCellMeta *m2 = lstm_cell_meta_alloc_buf(o, bias, bias_tape_start);
    for (int j = 0; j < fo; j++) {
      m2->muliw_vals[j] = muliw[j];
      m2->mulrw_vals[j] = mulrw[j];
    }
    for (int j = 0; j < o; j++) m2->prev_cell_vals[j] = pcell[j];

    /* Perturb bias */
    double saved = bias[k];
    bias[k] = saved + eps;
    lstm_cell_compute(m2, out_plus);
    bias[k] = saved - eps;
    lstm_cell_compute(m2, out_minus);
    bias[k] = saved;

    double num = 0.0;
    for (int j = 0; j < 2 * o; j++)
      num += (out_plus[j] - out_minus[j]) / (2.0 * eps);

    char name[64];
    snprintf(name, sizeof(name), "lstm_cell_buf_bwd d_bias[%d]", k);
    check_close(name, grad[bias_tape_start + k], num, 1e-5);
  }

  /* Also check mulIW gradients match non-buf path */
  for (int k = 0; k < fo; k++) {
    LstmCellMeta *m2 = lstm_cell_meta_alloc_buf(o, bias, bias_tape_start);
    for (int j = 0; j < fo; j++) {
      m2->muliw_vals[j] = muliw[j];
      m2->mulrw_vals[j] = mulrw[j];
    }
    for (int j = 0; j < o; j++) m2->prev_cell_vals[j] = pcell[j];

    m2->muliw_vals[k] = muliw[k] + eps;
    lstm_cell_compute(m2, out_plus);
    m2->muliw_vals[k] = muliw[k] - eps;
    lstm_cell_compute(m2, out_minus);

    double num = 0.0;
    for (int j = 0; j < 2 * o; j++)
      num += (out_plus[j] - out_minus[j]) / (2.0 * eps);

    char name[64];
    snprintf(name, sizeof(name), "lstm_cell_buf_bwd d_muliw[%d]", k);
    check_close(name, grad[k], num, 1e-5);
  }

  free(grad);
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
  test_batch_cossim_forward();
  test_batch_cossim_backward();
  test_readop_forward();
  test_readop_backward();
  test_writeop_forward();
  test_writeop_backward();
  test_interp_write_forward();
  test_interp_write_backward();
  test_ntm_mem_alloc();
  test_ntm_mem_set_val();
  test_ntm_mem_set_pid();
  test_ntm_mem_update_vals();
  test_ntm_mem_update_tape_idx();
  test_ntm_mem_sync_vals();
  test_ntm_mem_pack_batch_cossim();
  test_ntm_mem_pack_readop();
  test_ntm_mem_pack_interp_write();
  test_ntm_mem_full_lifecycle();
  test_interpolate_forward();
  test_interpolate_backward();
  test_shift_forward();
  test_shift_backward();
  test_focus_forward();
  test_focus_backward();
  test_lstm_cell_forward();
  test_lstm_cell_backward();
  test_matvec_bias_forward();
  test_matvec_bias_backward();
  test_lstm_cell_bias_buf_forward();
  test_lstm_cell_bias_buf_backward();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}

/* Operator-level benchmarks for the idris-ml C backend.
   Measures raw backend speed (no Idris/Chez overhead).

   Output format (one line per benchmark):
     label: X.XXX ms  (N iters)

   Build: cc -o build/bench_ops csrc/bench_ops.c -L build -lidrisml -Wl,-rpath,build -lm
   Run:   ./build/bench_ops
*/

#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ---------- Timing ---------- */

static double wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ---------- Helpers ---------- */

/* Create a 2D tensor [rows x cols] filled with small random-ish values */
static TensorHandle make_matrix(int rows, int cols, int requires_grad) {
    int n = rows * cols;
    double* data = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        data[i] = 0.01 * ((i * 7 + 13) % 100 - 50);  /* deterministic pseudo-random */
    int shape[2] = {rows, cols};
    TensorHandle t = tensor_create(data, shape, 2, requires_grad);
    free(data);
    return t;
}

/* Create a 1D tensor [n] filled with small values */
static TensorHandle make_vector(int n, int requires_grad) {
    double* data = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        data[i] = 0.01 * ((i * 3 + 7) % 100 - 50);
    int shape[] = {n};
    TensorHandle t = tensor_create(data, shape, 1, requires_grad);
    free(data);
    return t;
}

/* Create a 4D tensor [n, c, h, w] for conv2d input */
static TensorHandle make_4d(int n, int c, int h, int w, int requires_grad) {
    int numel = n * c * h * w;
    double* data = (double*)malloc(numel * sizeof(double));
    for (int i = 0; i < numel; i++)
        data[i] = 0.01 * ((i * 11 + 3) % 100 - 50);
    int shape[4] = {n, c, h, w};
    TensorHandle t = tensor_create(data, shape, 4, requires_grad);
    free(data);
    return t;
}

/* ================================================================
   Matmul benchmarks
   ================================================================ */

/* Force evaluation (needed for MLX lazy eval; harmless on tape/torch) */
static void force_eval(TensorHandle t) {
    volatile double v = tensor_item(tensor_sum(t));
    (void)v;
}

static void bench_matmul(int m, int n, int k, int iters) {
    TensorHandle a = make_matrix(m, n, 0);
    TensorHandle b = make_matrix(n, k, 0);

    /* warmup */
    for (int i = 0; i < 10; i++) {
        TensorHandle c = tensor_mm(a, b);
        force_eval(c);
        tensor_free(c);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle c = tensor_mm(a, b);
        force_eval(c);
        tensor_free(c);
    }
    double elapsed = wall_ms() - t0;

    printf("matmul %dx%dx%d:\t%.3f ms  (%d iters)\n", m, n, k, elapsed, iters);
    tensor_free(a);
    tensor_free(b);
}

/* ================================================================
   Matrix-vector benchmarks
   ================================================================ */

static void bench_matvec(int m, int n, int iters) {
    TensorHandle mat = make_matrix(m, n, 0);
    TensorHandle vec = make_vector(n, 0);

    for (int i = 0; i < 10; i++) {
        TensorHandle r = tensor_mv(mat, vec);
        force_eval(r);
        tensor_free(r);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle r = tensor_mv(mat, vec);
        force_eval(r);
        tensor_free(r);
    }
    double elapsed = wall_ms() - t0;

    printf("matvec %dx%d:\t%.3f ms  (%d iters)\n", m, n, elapsed, iters);
    tensor_free(mat);
    tensor_free(vec);
}

/* ================================================================
   Element-wise benchmarks (add + mul)
   ================================================================ */

static void bench_elementwise(int n, int iters) {
    TensorHandle a = make_vector(n, 0);
    TensorHandle b = make_vector(n, 0);

    for (int i = 0; i < 10; i++) {
        TensorHandle c = tensor_add(a, b);
        TensorHandle d = tensor_mul(c, b);
        force_eval(d);
        tensor_free(d);
        tensor_free(c);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle c = tensor_add(a, b);
        TensorHandle d = tensor_mul(c, b);
        force_eval(d);
        tensor_free(d);
        tensor_free(c);
    }
    double elapsed = wall_ms() - t0;

    printf("add+mul %d:\t%.3f ms  (%d iters)\n", n, elapsed, iters);
    tensor_free(a);
    tensor_free(b);
}

/* ================================================================
   Softmax benchmarks
   ================================================================ */

static void bench_softmax(int n, int iters) {
    TensorHandle a = make_vector(n, 0);

    for (int i = 0; i < 10; i++) {
        TensorHandle s = tensor_softmax(a, 0);
        force_eval(s);
        tensor_free(s);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle s = tensor_softmax(a, 0);
        force_eval(s);
        tensor_free(s);
    }
    double elapsed = wall_ms() - t0;

    printf("softmax %d:\t%.3f ms  (%d iters)\n", n, elapsed, iters);
    tensor_free(a);
}

/* ================================================================
   Conv2d forward benchmark
   ================================================================ */

static void bench_conv2d(int inC, int outC, int h, int w, int kH, int kW, int iters) {
    /* input: [1, inC, h, w] as flat [inC*h*w] */
    TensorHandle input = make_vector(inC * h * w, 0);
    TensorHandle kernel = make_4d(outC, inC, kH, kW, 0);
    TensorHandle bias = make_vector(outC, 0);

    for (int i = 0; i < 3; i++) {
        TensorHandle out = tensor_conv2d(input, kernel, bias, 0, 0, 1, 1);
        force_eval(out);
        tensor_free(out);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle out = tensor_conv2d(input, kernel, bias, 0, 0, 1, 1);
        force_eval(out);
        tensor_free(out);
    }
    double elapsed = wall_ms() - t0;

    int oH = h - kH + 1;
    int oW = w - kW + 1;
    printf("conv2d %dx%dx%d->%d k=%dx%d:\t%.3f ms  (%d iters)\n",
           inC, h, w, outC, kH, kW, elapsed, iters);
    tensor_free(input);
    tensor_free(kernel);
    tensor_free(bias);
}

/* ================================================================
   Forward + backward + step benchmark (single linear layer)
   ================================================================ */

static void bench_train_step(int inputDim, int outputDim, int iters) {
    /* Simple training step: y = Wx + b, loss = sum(y), backward, step.
       Uses tensor_create_param_2d / _1d which create persistent param tensors. */
    param_clear();

    /* Allocate param data on heap (tensor_create_param_* takes ownership) */
    double* wdata = (double*)malloc(outputDim * inputDim * sizeof(double));
    for (int i = 0; i < outputDim * inputDim; i++)
        wdata[i] = 0.01 * ((i * 7 + 13) % 100 - 50);
    double* bdata = (double*)malloc(outputDim * sizeof(double));
    for (int i = 0; i < outputDim; i++)
        bdata[i] = 0.0;

    TensorHandle W = tensor_create_param_2d(outputDim, inputDim, wdata);
    TensorHandle b = tensor_create_param_1d(outputDim, bdata);
    param_register("W", W);
    param_register("b", b);

    OptimizerHandle opt = optimizer_create_sgd(0.01);

    /* warmup */
    for (int i = 0; i < 5; i++) {
        TensorHandle x = make_vector(inputDim, 0);
        TensorHandle y = tensor_add(tensor_mv(W, x), b);
        TensorHandle loss = tensor_sum(y);
        tensor_backward(loss);
        optimizer_step(opt);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle x = make_vector(inputDim, 0);
        TensorHandle y = tensor_add(tensor_mv(W, x), b);
        TensorHandle loss = tensor_sum(y);
        tensor_backward(loss);
        optimizer_step(opt);
    }
    double elapsed = wall_ms() - t0;

    printf("train_step %d->%d:\t%.3f ms  (%d iters)\n",
           inputDim, outputDim, elapsed, iters);
    optimizer_free(opt);
}

/* ================================================================
   Main
   ================================================================ */

int main(void) {
    /* Warmup: trigger arena + tape initialization so it doesn't pollute benchmarks */
    {
        TensorHandle a = tensor_create_scalar(1.0, 1);
        TensorHandle b = tensor_create_scalar(2.0, 0);
        TensorHandle c = tensor_add(a, b);
        TensorHandle loss = tensor_sum(c);
        tensor_backward(loss);
        backend_reset_for_eval();
    }

    printf("=== Operator Benchmarks (C backend) ===\n\n");

    /* --- Matmul --- */
    printf("--- Matrix multiply ---\n");
    bench_matmul(64, 64, 64, 500);
    bench_matmul(256, 256, 256, 100);
    bench_matmul(1024, 1024, 1024, 10);
    fflush(stdout);
    printf("\n");

    /* --- Matrix-vector --- */
    printf("--- Matrix-vector multiply ---\n");
    bench_matvec(256, 256, 1000);
    bench_matvec(1024, 1024, 200);
    fflush(stdout);
    printf("\n");

    /* --- Element-wise --- */
    printf("--- Element-wise (add + mul) ---\n");
    bench_elementwise(1000, 1000);
    bench_elementwise(10000, 500);
    bench_elementwise(100000, 100);
    fflush(stdout);
    printf("\n");

    /* --- Softmax --- */
    printf("--- Softmax ---\n");
    bench_softmax(256, 1000);
    bench_softmax(1024, 500);
    bench_softmax(10000, 100);
    fflush(stdout);
    printf("\n");

    /* --- Training step (forward + backward + optimizer) --- */
    printf("--- Training step (linear fwd+bwd+step) ---\n");
    bench_train_step(64, 64, 200);
    bench_train_step(256, 256, 100);
    bench_train_step(1024, 1024, 10);
    fflush(stdout);
    printf("\n");

    /* --- Conv2d forward (last: may crash on torch backend due to shape mismatch) --- */
    printf("--- Conv2d forward ---\n");
    bench_conv2d(1, 16, 28, 28, 5, 5, 10);      /* MNIST layer 1 */
    bench_conv2d(16, 32, 12, 12, 5, 5, 10);     /* MNIST layer 2 */
    fflush(stdout);
    printf("\n");

    printf("=== Done ===\n");
    return 0;
}

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
   Scaled-dot-product attention (Axis A: kernel bench)
   ================================================================ */

/* SDPA bench - representative of HF Llama-class decoder attention with GQA.
 *   Q : [seq, numHeads   * headDim]
 *   K : [seq, numKvHeads * headDim]
 *   V : [seq, numKvHeads * headDim]
 *   out [seq, numHeads * headDim]
 */
static void bench_attention_sdpa(int seq, int numHeads, int numKvHeads,
                                 int headDim, int isCausal, int iters) {
    TensorHandle q = make_matrix(seq, numHeads   * headDim, 0);
    TensorHandle k = make_matrix(seq, numKvHeads * headDim, 0);
    TensorHandle v = make_matrix(seq, numKvHeads * headDim, 0);

    for (int i = 0; i < 5; i++) {
        TensorHandle o = tensor_sdpa_2d(q, k, v, numHeads, numKvHeads, headDim, isCausal);
        force_eval(o);
        tensor_free(o);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle o = tensor_sdpa_2d(q, k, v, numHeads, numKvHeads, headDim, isCausal);
        force_eval(o);
        tensor_free(o);
    }
    double elapsed = wall_ms() - t0;

    printf("sdpa seq=%d H=%d Hkv=%d d=%d%s:\t%.3f ms  (%d iters)\n",
           seq, numHeads, numKvHeads, headDim, isCausal ? " causal" : "",
           elapsed, iters);
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
}

/* ================================================================
   Embedding gather (Axis A: kernel bench)
   ================================================================ */

/* Embedding lookup bench - representative of transformer input embedding.
 *   weight  [vocabSize, embedDim]
 *   indices [n] (integer-valued doubles)
 *   out     [n, embedDim]
 */
static void bench_embedding_gather(int vocabSize, int embedDim, int n, int iters) {
    TensorHandle weight = make_matrix(vocabSize, embedDim, 0);

    /* Deterministic int-valued indices in [0, vocabSize). Use unsigned
       arithmetic to avoid signed-integer-overflow UB at large multipliers. */
    double* idata = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        unsigned long h = (unsigned long)i * 2654435761ul + 12345ul;
        idata[i] = (double)(h % (unsigned long)vocabSize);
    }
    int ishape[] = {n};
    TensorHandle indices = tensor_create(idata, ishape, 1, 0);
    free(idata);

    for (int i = 0; i < 5; i++) {
        TensorHandle o = tensor_embedding_2d(weight, indices, n, embedDim);
        force_eval(o);
        tensor_free(o);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle o = tensor_embedding_2d(weight, indices, n, embedDim);
        force_eval(o);
        tensor_free(o);
    }
    double elapsed = wall_ms() - t0;

    printf("embedding vocab=%d d=%d n=%d:\t%.3f ms  (%d iters)\n",
           vocabSize, embedDim, n, elapsed, iters);
    tensor_free(weight);
    tensor_free(indices);
}

/* ================================================================
   Fused RMSNorm (Axis A: kernel bench)
   ================================================================ */

/* Fused RMSNorm bench - the per-block normalization in HF Llama / Mistral.
 *   input  [seqLen, hidden]
 *   weight [hidden]
 *   out    [seqLen, hidden]
 */
static void bench_rms_norm(int seqLen, int hidden, int iters) {
    TensorHandle input = make_matrix(seqLen, hidden, 0);
    TensorHandle weight = make_vector(hidden, 0);

    for (int i = 0; i < 5; i++) {
        TensorHandle o = tensor_rms_norm_2d(input, weight, 1e-6);
        force_eval(o);
        tensor_free(o);
    }

    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        TensorHandle o = tensor_rms_norm_2d(input, weight, 1e-6);
        force_eval(o);
        tensor_free(o);
    }
    double elapsed = wall_ms() - t0;

    printf("rmsnorm seq=%d h=%d:\t%.3f ms  (%d iters)\n",
           seqLen, hidden, elapsed, iters);
    tensor_free(input);
    tensor_free(weight);
}

/* ================================================================
   Conv2d forward benchmark
   ================================================================ */

static void bench_conv2d(int inC, int outC, int h, int w, int kH, int kW, int iters) {
    /* tensor_conv2d (single-sample) takes a rank-3 input [inC, H, W].
       Passing a flat rank-1 buffer reads past shape[] into uninitialised
       memory for H/W and either loops over garbage or segfaults — that
       was the original bench_ops crash before the rank fix landed. */
    int numel = inC * h * w;
    double* idata = (double*)malloc(numel * sizeof(double));
    for (int i = 0; i < numel; i++)
        idata[i] = 0.01 * ((i * 3 + 7) % 100 - 50);
    int ishape[3] = {inC, h, w};
    TensorHandle input = tensor_create(idata, ishape, 3, 0);
    free(idata);
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
       Uses tensor_create_param_{2d,1d}_f64 which create persistent param tensors. */
    param_clear();

    /* Allocate param data on heap (tensor_create_param_* takes ownership) */
    double* wdata = (double*)malloc(outputDim * inputDim * sizeof(double));
    for (int i = 0; i < outputDim * inputDim; i++)
        wdata[i] = 0.01 * ((i * 7 + 13) % 100 - 50);
    double* bdata = (double*)malloc(outputDim * sizeof(double));
    for (int i = 0; i < outputDim; i++)
        bdata[i] = 0.0;

    TensorHandle W = tensor_create_param_2d_f64(outputDim, inputDim, wdata);
    TensorHandle b = tensor_create_param_1d_f64(outputDim, bdata);
    param_register("W", W);
    param_register("b", b);

    OptimizerHandle opt = optimizer_create_sgd(0.01);

    /* warmup */
    for (int i = 0; i < 5; i++) {
        backend_epoch_begin();
        TensorHandle x = make_vector(inputDim, 0);
        TensorHandle y = tensor_linear(W, x, b);
        TensorHandle loss = tensor_sum(y);
        tensor_backward(loss);
        optimizer_step(opt);
    }

    backend_profile_reset();
    double t0 = wall_ms();
    for (int i = 0; i < iters; i++) {
        backend_epoch_begin();
        TensorHandle x = make_vector(inputDim, 0);
        TensorHandle y = tensor_linear(W, x, b);
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

    backend_profile_report();

    /* --- Conv2d forward --- */
    printf("--- Conv2d forward ---\n");
    bench_conv2d(1, 16, 28, 28, 5, 5, 10);
    bench_conv2d(16, 32, 12, 12, 5, 5, 10);
    fflush(stdout);
    printf("\n");

    /* Reset tape + param registry once before Axis A so the SDPA /
       embedding / rmsnorm benches start from a clean tape. */
    backend_reset_for_eval();

    /* --- Scaled-dot-product attention (Axis A) --- */
    printf("--- Scaled-dot-product attention ---\n");
    /* Mini-Llama-class GQA (8 query heads, 4 KV heads, headDim=64). */
    bench_attention_sdpa(64,  8, 4, 64, 0, 100);
    bench_attention_sdpa(128, 8, 4, 64, 0, 50);
    bench_attention_sdpa(128, 8, 4, 64, 1, 50);    /* causal */
    fflush(stdout);
    printf("\n");

    /* --- Embedding gather (Axis A) --- */
    printf("--- Embedding gather ---\n");
    bench_embedding_gather(32000, 128, 128, 200);   /* GPT-2-class vocab */
    bench_embedding_gather(8000,  256, 64,  500);   /* mid-vocab decoder */
    fflush(stdout);
    printf("\n");

    /* --- RMSNorm fused (Axis A) --- */
    printf("--- RMSNorm fused ---\n");
    bench_rms_norm(128, 512, 500);
    bench_rms_norm(128, 2048, 100);
    fflush(stdout);
    printf("\n");

    printf("=== Done ===\n");
    return 0;
}

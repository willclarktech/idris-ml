/* backend_tape.c — Tape-based autograd backend implementing backend.h.
 *
 * Each tensor is a scalar double with an index into a global tape.
 * Multi-dimensional tensors are flat double arrays with shape metadata.
 * Forward ops append to the tape; backward walks it in reverse.
 *
 * Design: arena-allocated tape + Accelerate BLAS for linalg.
 */

#include "backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <sys/resource.h>
#include <mach/mach.h>
#endif

/* ================================================================
   Tensor representation
   ================================================================ */

typedef struct {
    double* data;       /* owned, heap-allocated */
    int* shape;         /* owned, heap-allocated (NULL for scalar) */
    int rank;           /* 0 = scalar, 1 = vector, 2 = matrix */
    int numel;
    int requires_grad;
    int tape_idx;       /* index into tape (-1 if not tracked) */
    double* grad;       /* gradient storage (same shape as data), NULL if not allocated */
    int persistent;     /* 1 = param tensor (malloc'd), 0 = intermediate (arena) */
} Tensor;

/* ================================================================
   Arena allocator — bump-pointer for intermediate tensors.
   Reset in bulk at optimizer_step. Params use regular malloc.
   ================================================================ */

#define ARENA_INIT_SIZE (1 << 20)  /* 1 MB */

typedef struct ArenaChunk {
    char* data;
    size_t cap;
    size_t used;
    struct ArenaChunk* next;
} ArenaChunk;

static ArenaChunk* arena_head = NULL;
static ArenaChunk* arena_current = NULL;

static ArenaChunk* arena_new_chunk(size_t min_size) {
    size_t cap = min_size > ARENA_INIT_SIZE ? min_size : ARENA_INIT_SIZE;
    ArenaChunk* c = malloc(sizeof(ArenaChunk));
    c->data = malloc(cap);
    c->cap = cap;
    c->used = 0;
    c->next = NULL;
    return c;
}

static void* arena_alloc(size_t bytes) {
    /* Align to 8 bytes */
    bytes = (bytes + 7) & ~7;
    if (!arena_head) {
        arena_head = arena_new_chunk(ARENA_INIT_SIZE);
        arena_current = arena_head;
    }
    if (arena_current->used + bytes > arena_current->cap) {
        /* Need new chunk */
        if (arena_current->next) {
            arena_current = arena_current->next;
            arena_current->used = 0;
        } else {
            ArenaChunk* c = arena_new_chunk(bytes);
            arena_current->next = c;
            arena_current = c;
        }
    }
    void* ptr = arena_current->data + arena_current->used;
    arena_current->used += bytes;
    return ptr;
}

static void arena_reset(void) {
    ArenaChunk* c = arena_head;
    while (c) { c->used = 0; c = c->next; }
    arena_current = arena_head;
}

/* make_scalar/make_tensor: use arena for intermediate tensors */

static Tensor* make_scalar(double val, int requires_grad) {
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    double* d = arena_alloc(sizeof(double));
    d[0] = val;
    t->data = d;
    t->shape = NULL;
    t->rank = 0;
    t->numel = 1;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->grad = NULL;
    t->persistent = 0;
    return t;
}

static Tensor* make_tensor(double* data, int* shape, int rank, int requires_grad) {
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    t->data = arena_alloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    t->shape = arena_alloc(rank * sizeof(int));
    memcpy(t->shape, shape, rank * sizeof(int));
    t->rank = rank;
    t->numel = numel;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->grad = NULL;
    t->persistent = 0;
    return t;
}

static void ensure_grad(Tensor* t) {
    if (!t->grad) {
        t->grad = calloc(t->numel, sizeof(double));
    }
}

/* ================================================================
   Tape
   ================================================================ */

/* Operation tags */
enum {
    OP_CONST = 0,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_NEG, OP_ABS, OP_EXP, OP_LOG, OP_SQRT, OP_POW,
    OP_SIGMOID, OP_TANH,
    OP_MV, OP_DOT, OP_OUTER,
    OP_SOFTMAX, OP_LOG_SOFTMAX,
    OP_SUM, OP_MEAN,
    OP_BCE_WITH_LOGITS,
    OP_NTM_READ_HEAD, OP_NTM_INTERP_WRITE,
    OP_LSTM_GATES,
    OP_ADD_SCALAR, OP_MUL_SCALAR, OP_CLAMP_MIN,
    OP_COSINE_SIM, OP_CONV1D_CIRC,
    OP_STACK,     /* stack of scalar tensors into 1D */
    OP_RESHAPE,   /* reshape (view) — grad passes through unchanged */
    OP_SELECT,    /* select element from vector — grad goes to parent[index] */
};

typedef struct {
    int op;
    Tensor* result;     /* non-owning pointer to the result tensor */
    Tensor* arg1;       /* non-owning: first input */
    Tensor* arg2;       /* non-owning: second input (NULL for unary) */
    double scalar_arg;  /* for add_scalar, mul_scalar */
    Tensor** inputs;    /* for OP_STACK: array of constituent scalar tensors */
    int input_count;    /* number of inputs for stack */
} TapeEntry;

#define TAPE_INIT_CAP 4096

static TapeEntry* tape = NULL;
static int tape_size = 0;
static int tape_cap = 0;

static void tape_init(void) {
    if (!tape) {
        tape_cap = TAPE_INIT_CAP;
        tape = calloc(tape_cap, sizeof(TapeEntry));
        tape_size = 0;
    }
}

static int tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
    tape_init();
    if (tape_size >= tape_cap) {
        tape_cap *= 2;
        tape = realloc(tape, tape_cap * sizeof(TapeEntry));
    }
    int idx = tape_size++;
    tape[idx].op = op;
    tape[idx].result = result;
    tape[idx].arg1 = arg1;
    tape[idx].arg2 = arg2;
    tape[idx].scalar_arg = scalar_arg;
    result->tape_idx = idx;
    return idx;
}

static void tape_reset(void) {
    tape_size = 0;
}

/* ================================================================
   Lifecycle
   ================================================================ */

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    Tensor* t;
    if (requires_grad) {
        /* Param tensors use regular malloc — persist across arena resets */
        t = calloc(1, sizeof(Tensor));
        t->data = malloc(sizeof(double));
        t->data[0] = value;
        t->rank = 0; t->numel = 1;
        t->requires_grad = 1;
        t->tape_idx = -1;
        t->persistent = 1;
        tape_append(OP_CONST, t, NULL, NULL, 0);
    } else {
        t = make_scalar(value, 0);
    }
    return t;
}

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    Tensor* t = make_tensor(data, shape, rank, requires_grad);
    if (requires_grad) {
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
    return t;
}

TensorHandle tensor_clone(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (t->rank == 0) return make_scalar(t->data[0], 0);
    return make_tensor(t->data, t->shape, t->rank, 0);
}

void tensor_free(TensorHandle h) {
    /* Note: we don't actually free during a tape lifetime since the tape
       holds non-owning pointers. In practice, tensors live until tape_reset.
       For now, this is a no-op to avoid use-after-free. */
    (void)h;
}

/* ================================================================
   Accessors
   ================================================================ */

double tensor_item(TensorHandle h) {
    return ((Tensor*)h)->data[0];
}

int tensor_numel(TensorHandle h) {
    return ((Tensor*)h)->numel;
}

int tensor_dim(TensorHandle h) {
    return ((Tensor*)h)->rank;
}

int tensor_size(TensorHandle h, int dim) {
    Tensor* t = (Tensor*)h;
    if (dim < t->rank) return t->shape[dim];
    return 0;
}

void tensor_to_doubles(TensorHandle h, double* out) {
    Tensor* t = (Tensor*)h;
    memcpy(out, t->data, t->numel * sizeof(double));
}

/* ================================================================
   Scalar arithmetic (forward only — backward in tape walk)
   ================================================================ */

#define SCALAR_BINOP(name, op_tag, expr) \
TensorHandle name(TensorHandle ha, TensorHandle hb) { \
    Tensor* a = (Tensor*)ha; Tensor* b = (Tensor*)hb; \
    double val = (expr); \
    Tensor* r = make_scalar(val, a->requires_grad || b->requires_grad); \
    if (r->requires_grad) tape_append(op_tag, r, a, b, 0); \
    return r; \
}

#define SCALAR_UNOP(name, op_tag, expr) \
TensorHandle name(TensorHandle ha) { \
    Tensor* a = (Tensor*)ha; \
    double val = (expr); \
    Tensor* r = make_scalar(val, a->requires_grad); \
    if (r->requires_grad) tape_append(op_tag, r, a, NULL, 0); \
    return r; \
}

/* Element-wise binary ops: handle both scalar and multi-dim */
static TensorHandle binop_elementwise(TensorHandle ha, TensorHandle hb, int op_tag,
                                       double (*scalar_fn)(double, double)) {
    Tensor* a = (Tensor*)ha; Tensor* b = (Tensor*)hb;
    int rg = a->requires_grad || b->requires_grad;
    if (a->numel == 1 && b->numel == 1) {
        Tensor* r = make_scalar(scalar_fn(a->data[0], b->data[0]), rg);
        if (rg) tape_append(op_tag, r, a, b, 0);
        return r;
    }
    /* Multi-dim: element-wise with broadcasting */
    int n = a->numel > b->numel ? a->numel : b->numel;
    double* data = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double av = a->data[a->numel == 1 ? 0 : i];
        double bv = b->data[b->numel == 1 ? 0 : i];
        data[i] = scalar_fn(av, bv);
    }
    Tensor* big = a->numel >= b->numel ? a : b;
    Tensor* r = make_tensor(data, big->shape, big->rank, rg);
    free(data);
    if (rg) tape_append(op_tag, r, a, b, 0);
    return r;
}

static double fn_add(double a, double b) { return a + b; }
static double fn_sub(double a, double b) { return a - b; }
static double fn_mul(double a, double b) { return a * b; }
static double fn_div(double a, double b) { return a / b; }
static double fn_pow(double a, double b) { return pow(a, b); }

TensorHandle tensor_add(TensorHandle a, TensorHandle b) { return binop_elementwise(a, b, OP_ADD, fn_add); }
TensorHandle tensor_sub(TensorHandle a, TensorHandle b) { return binop_elementwise(a, b, OP_SUB, fn_sub); }
TensorHandle tensor_mul(TensorHandle a, TensorHandle b) { return binop_elementwise(a, b, OP_MUL, fn_mul); }
TensorHandle tensor_div(TensorHandle a, TensorHandle b) { return binop_elementwise(a, b, OP_DIV, fn_div); }
TensorHandle tensor_pow(TensorHandle a, TensorHandle b) { return binop_elementwise(a, b, OP_POW, fn_pow); }

SCALAR_UNOP(tensor_neg, OP_NEG, -a->data[0])
SCALAR_UNOP(tensor_abs, OP_ABS, fabs(a->data[0]))
SCALAR_UNOP(tensor_exp, OP_EXP, exp(a->data[0]))
SCALAR_UNOP(tensor_log, OP_LOG, log(a->data[0]))
SCALAR_UNOP(tensor_sqrt, OP_SQRT, sqrt(a->data[0]))

TensorHandle tensor_sigmoid(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    double val = 1.0 / (1.0 + exp(-a->data[0]));
    Tensor* r = make_scalar(val, a->requires_grad);
    if (r->requires_grad) tape_append(OP_SIGMOID, r, a, NULL, 0);
    return r;
}

TensorHandle tensor_tanh(TensorHandle ha) {
    Tensor* a = (Tensor*)ha;
    double val = tanh(a->data[0]);
    Tensor* r = make_scalar(val, a->requires_grad);
    if (r->requires_grad) tape_append(OP_TANH, r, a, NULL, 0);
    return r;
}

TensorHandle tensor_add_scalar(TensorHandle ha, double s) {
    Tensor* a = (Tensor*)ha;
    if (a->numel == 1) {
        Tensor* r = make_scalar(a->data[0] + s, a->requires_grad);
        if (r->requires_grad) tape_append(OP_ADD_SCALAR, r, a, NULL, s);
        return r;
    }
    /* Multi-element: add scalar to each element */
    double* data = arena_alloc(a->numel * sizeof(double));
    for (int i = 0; i < a->numel; i++) data[i] = a->data[i] + s;
    Tensor* r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    if (r->requires_grad) tape_append(OP_ADD_SCALAR, r, a, NULL, s);
    return r;
}

TensorHandle tensor_mul_scalar(TensorHandle ha, double s) {
    Tensor* a = (Tensor*)ha;
    if (a->numel == 1) {
        Tensor* r = make_scalar(a->data[0] * s, a->requires_grad);
        if (r->requires_grad) tape_append(OP_MUL_SCALAR, r, a, NULL, s);
        return r;
    }
    /* Multi-element: multiply each element by scalar */
    double* data = arena_alloc(a->numel * sizeof(double));
    for (int i = 0; i < a->numel; i++) data[i] = a->data[i] * s;
    Tensor* r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    if (r->requires_grad) tape_append(OP_MUL_SCALAR, r, a, NULL, s);
    return r;
}

TensorHandle tensor_clamp_min(TensorHandle ha, double min_val) {
    Tensor* a = (Tensor*)ha;
    int n = a->numel;
    double* data = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) data[i] = fmax(a->data[i], min_val);
    Tensor* r;
    if (n == 1) {
        r = make_scalar(data[0], a->requires_grad);
    } else {
        r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    }
    free(data);
    if (r->requires_grad) tape_append(OP_CLAMP_MIN, r, a, NULL, min_val);
    return r;
}

/* ================================================================
   Reduction
   ================================================================ */

TensorHandle tensor_sum(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    double s = 0;
    for (int i = 0; i < t->numel; i++) s += t->data[i];
    Tensor* r = make_scalar(s, t->requires_grad);
    if (r->requires_grad) tape_append(OP_SUM, r, t, NULL, 0);
    return r;
}

TensorHandle tensor_sum_dim(TensorHandle h, int dim, int keepdim) {
    /* Simplified: only support full sum for now */
    return tensor_sum(h);
}

TensorHandle tensor_mean(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    double s = 0;
    for (int i = 0; i < t->numel; i++) s += t->data[i];
    Tensor* r = make_scalar(s / t->numel, t->requires_grad);
    if (r->requires_grad) tape_append(OP_MEAN, r, t, NULL, 0);
    return r;
}

/* ================================================================
   Linear algebra
   ================================================================ */

TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
    Tensor* mat = (Tensor*)hmat;
    Tensor* vec = (Tensor*)hvec;
    int m = mat->shape[0], n = mat->shape[1];
    int out_shape[] = {m};
    double* out_data = calloc(m, sizeof(double));

#ifdef __APPLE__
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
                mat->data, n, vec->data, 1, 0.0, out_data, 1);
#else
    for (int i = 0; i < m; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += mat->data[i*n+j] * vec->data[j];
        out_data[i] = s;
    }
#endif

    Tensor* r = make_tensor(out_data, out_shape, 1, mat->requires_grad || vec->requires_grad);
    free(out_data);
    if (r->requires_grad) tape_append(OP_MV, r, mat, vec, 0);
    return r;
}

TensorHandle tensor_dot(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    double s = 0;
    for (int i = 0; i < a->numel; i++) s += a->data[i] * b->data[i];
    Tensor* r = make_scalar(s, a->requires_grad || b->requires_grad);
    if (r->requires_grad) tape_append(OP_DOT, r, a, b, 0);
    return r;
}

TensorHandle tensor_matmul(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    /* 1D x 2D = mv transpose, or delegate based on ranks */
    if (a->rank == 1 && b->rank == 2) {
        /* [n] x [n,m] = [m] — treat as row vector matmul */
        int n = a->numel, m = b->shape[1];
        int out_shape[] = {m};
        double* out_data = calloc(m, sizeof(double));
        for (int j = 0; j < m; j++) {
            double s = 0;
            for (int i = 0; i < n; i++) s += a->data[i] * b->data[i*m+j];
            out_data[j] = s;
        }
        Tensor* r = make_tensor(out_data, out_shape, 1, a->requires_grad || b->requires_grad);
        free(out_data);
        if (r->requires_grad) tape_append(OP_DOT, r, a, b, 0);
        return r;
    }
    if (a->rank == 2 && b->rank == 1) return tensor_mv(ha, hb);
    /* Fallback: scalar mul */
    return tensor_mul(ha, hb);
}

TensorHandle tensor_outer(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    int m = a->numel, n = b->numel;
    int shape[] = {m, n};
    double* data = malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            data[i*n+j] = a->data[i] * b->data[j];
    Tensor* r = make_tensor(data, shape, 2, a->requires_grad || b->requires_grad);
    free(data);
    if (r->requires_grad) tape_append(OP_OUTER, r, a, b, 0);
    return r;
}

/* ================================================================
   Activation / normalization
   ================================================================ */

TensorHandle tensor_softmax(TensorHandle h, int dim) {
    Tensor* t = (Tensor*)h;
    int n = t->numel;
    double* data = malloc(n * sizeof(double));
    double max_val = t->data[0];
    for (int i = 1; i < n; i++) if (t->data[i] > max_val) max_val = t->data[i];
    double sum = 0;
    for (int i = 0; i < n; i++) { data[i] = exp(t->data[i] - max_val); sum += data[i]; }
    for (int i = 0; i < n; i++) data[i] /= sum;
    Tensor* r;
    if (t->rank == 0) {
        r = make_scalar(data[0], t->requires_grad);
    } else {
        r = make_tensor(data, t->shape, t->rank, t->requires_grad);
    }
    free(data);
    if (r->requires_grad) tape_append(OP_SOFTMAX, r, t, NULL, 0);
    return r;
}

TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
    Tensor* t = (Tensor*)h;
    int n = t->numel;
    double* data = malloc(n * sizeof(double));
    double max_val = t->data[0];
    for (int i = 1; i < n; i++) if (t->data[i] > max_val) max_val = t->data[i];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += exp(t->data[i] - max_val);
    double log_sum = log(sum) + max_val;
    for (int i = 0; i < n; i++) data[i] = t->data[i] - log_sum;
    Tensor* r;
    if (t->rank == 0) {
        r = make_scalar(data[0], t->requires_grad);
    } else {
        r = make_tensor(data, t->shape, t->rank, t->requires_grad);
    }
    free(data);
    if (r->requires_grad) tape_append(OP_LOG_SOFTMAX, r, t, NULL, 0);
    return r;
}

/* ================================================================
   Loss functions
   ================================================================ */

TensorHandle tensor_bce_with_logits(TensorHandle hinput, TensorHandle htarget) {
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    int n = input->numel;
    double loss = 0;
    for (int i = 0; i < n; i++) {
        double p = input->data[i], y = target->data[i];
        double max_p = p > 0 ? p : 0;
        loss += max_p - p * y + log(1.0 + exp(-fabs(p)));
    }
    loss /= n;
    Tensor* r = make_scalar(loss, input->requires_grad);
    if (r->requires_grad) tape_append(OP_BCE_WITH_LOGITS, r, input, target, 0);
    return r;
}

TensorHandle tensor_cross_entropy(TensorHandle hinput, TensorHandle htarget) {
    /* Simplified: compute -sum(target * log_softmax(input)) / n */
    TensorHandle ls = tensor_log_softmax(hinput, 0);
    Tensor* lsT = (Tensor*)ls;
    Tensor* target = (Tensor*)htarget;
    double loss = 0;
    for (int i = 0; i < lsT->numel; i++) loss -= target->data[i] * lsT->data[i];
    loss /= lsT->numel;
    return make_scalar(loss, 0); /* simplified, no grad */
}

TensorHandle tensor_mse_loss(TensorHandle hinput, TensorHandle htarget) {
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    double loss = 0;
    for (int i = 0; i < input->numel; i++) {
        double d = input->data[i] - target->data[i];
        loss += d * d;
    }
    return make_scalar(loss / input->numel, 0);
}

/* ================================================================
   NTM-specific compositions
   ================================================================ */

TensorHandle tensor_cosine_similarity(TensorHandle ha, TensorHandle hb, int dim) {
    /* Simplified: compute row-wise cosine sim of a [n,w] vs b [1,w] */
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    if (a->rank == 2 && b->rank == 2) {
        int n = a->shape[0], w = a->shape[1];
        int out_shape[] = {n};
        double* out = calloc(n, sizeof(double));
        double* brow = b->data; /* [1, w] -> just use first row */
        double bnorm = 0;
        for (int j = 0; j < w; j++) bnorm += brow[j] * brow[j];
        bnorm = sqrt(bnorm) + 1e-8;
        for (int i = 0; i < n; i++) {
            double dot = 0, anorm = 0;
            for (int j = 0; j < w; j++) {
                dot += a->data[i*w+j] * brow[j];
                anorm += a->data[i*w+j] * a->data[i*w+j];
            }
            anorm = sqrt(anorm) + 1e-8;
            out[i] = dot / (anorm * bnorm);
        }
        Tensor* r = make_tensor(out, out_shape, 1, a->requires_grad || b->requires_grad);
        free(out);
        if (r->requires_grad) tape_append(OP_COSINE_SIM, r, a, b, 0);
        return r;
    }
    return make_scalar(0, 0); /* fallback */
}

TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    int n = input->numel, k = kernel->numel, pad = k / 2;
    double* out = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < k; j++) {
            int idx = (i - pad + j + n) % n;
            s += input->data[idx] * kernel->data[k - 1 - j];
        }
        out[i] = s;
    }
    int shape[] = {n};
    Tensor* r = make_tensor(out, shape, 1, input->requires_grad || kernel->requires_grad);
    free(out);
    if (r->requires_grad) tape_append(OP_CONV1D_CIRC, r, input, kernel, 0);
    return r;
}

TensorPair* tensor_ntm_read_head(
    TensorHandle memory_h, TensorHandle prev_weights_h,
    TensorHandle key_h, TensorHandle beta_h, TensorHandle g_h,
    TensorHandle gamma_h, TensorHandle shift_kernel_h)
{
    Tensor* memory = (Tensor*)memory_h;
    Tensor* prev_w = (Tensor*)prev_weights_h;
    Tensor* key = (Tensor*)key_h;
    Tensor* beta = (Tensor*)beta_h;
    Tensor* g_t = (Tensor*)g_h;
    Tensor* gamma = (Tensor*)gamma_h;
    Tensor* shift = (Tensor*)shift_kernel_h;

    int n = memory->shape[0], w = memory->shape[1];
    double beta_v = beta->data[0], g_v = g_t->data[0], gamma_v = gamma->data[0];

    /* 1. Content addressing */
    double* cos_sim = calloc(n, sizeof(double));
    double key_norm = 0;
    for (int j = 0; j < w; j++) key_norm += key->data[j] * key->data[j];
    key_norm = sqrt(key_norm) + 1e-8;
    for (int i = 0; i < n; i++) {
        double dot = 0, row_norm = 0;
        for (int j = 0; j < w; j++) {
            dot += memory->data[i*w+j] * key->data[j];
            row_norm += memory->data[i*w+j] * memory->data[i*w+j];
        }
        row_norm = sqrt(row_norm) + 1e-8;
        cos_sim[i] = beta_v * dot / (row_norm * key_norm);
    }
    /* softmax */
    double max_cs = cos_sim[0];
    for (int i = 1; i < n; i++) if (cos_sim[i] > max_cs) max_cs = cos_sim[i];
    double sum_exp = 0;
    for (int i = 0; i < n; i++) { cos_sim[i] = exp(cos_sim[i] - max_cs); sum_exp += cos_sim[i]; }
    for (int i = 0; i < n; i++) cos_sim[i] /= sum_exp;

    /* 2. Interpolation */
    double* interp = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        interp[i] = g_v * cos_sim[i] + (1.0 - g_v) * prev_w->data[i];

    /* 3. Circular shift */
    int k = shift->numel, pad = k / 2;
    double* shifted = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < k; j++) {
            int idx = (i - pad + j + n) % n;
            s += interp[idx] * shift->data[k - 1 - j];
        }
        shifted[i] = s;
    }

    /* 4. Sharpening */
    double* focused = calloc(n, sizeof(double));
    double pow_sum = 0;
    for (int i = 0; i < n; i++) {
        focused[i] = pow(fmax(shifted[i], 1e-10), gamma_v);
        pow_sum += focused[i];
    }
    for (int i = 0; i < n; i++) focused[i] /= (pow_sum + 1e-10);

    /* 5. Read */
    double* read_out = calloc(w, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < w; j++)
            read_out[j] += focused[i] * memory->data[i*w+j];

    int w_shape[] = {n};
    int r_shape[] = {w};
    TensorPair* pair = malloc(sizeof(TensorPair));
    pair->first = make_tensor(focused, w_shape, 1, 0);
    pair->second = make_tensor(read_out, r_shape, 1, 0);

    free(cos_sim); free(interp); free(shifted); free(focused); free(read_out);
    return pair;
}

TensorHandle tensor_ntm_interp_write(
    TensorHandle memory_h, TensorHandle weights_h, TensorHandle add_vector_h)
{
    Tensor* memory = (Tensor*)memory_h;
    Tensor* weights = (Tensor*)weights_h;
    Tensor* add_vec = (Tensor*)add_vector_h;
    int n = memory->shape[0], w = memory->shape[1];
    double* data = malloc(n * w * sizeof(double));
    memcpy(data, memory->data, n * w * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < w; j++)
            data[i*w+j] += weights->data[i] * add_vec->data[j];
    int shape[] = {n, w};
    Tensor* r = make_tensor(data, shape, 2, 0);
    free(data);
    return r;
}

/* ================================================================
   Shape manipulation
   ================================================================ */

TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    Tensor* t = (Tensor*)h;
    /* Create a new tensor with different shape but shared data */
    Tensor* r = calloc(1, sizeof(Tensor));
    r->data = t->data;  /* shared */
    r->shape = malloc(rank * sizeof(int));
    memcpy(r->shape, shape, rank * sizeof(int));
    r->rank = rank;
    r->numel = t->numel;
    r->requires_grad = t->requires_grad;
    r->tape_idx = -1;
    r->grad = NULL;
    if (r->requires_grad) tape_append(OP_RESHAPE, r, t, NULL, 0);
    return r;
}

TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
    Tensor* t = (Tensor*)h;
    if (t->rank == 1) {
        int shape[] = {1, t->numel};
        return tensor_reshape(h, shape, 2);  /* shares data + records OP_RESHAPE */
    }
    return tensor_clone(h);
}

TensorHandle tensor_squeeze(TensorHandle h, int dim) {
    return tensor_clone(h); /* simplified */
}

TensorHandle tensor_select(TensorHandle h, int dim, int index) {
    Tensor* t = (Tensor*)h;
    if (t->rank == 1) {
        Tensor* v = calloc(1, sizeof(Tensor));
        v->data = &t->data[index];
        v->shape = NULL;
        v->rank = 0;
        v->numel = 1;
        v->requires_grad = t->requires_grad;
        v->tape_idx = -1;
        v->grad = NULL;
        /* Record OP_SELECT so backward propagates grad to parent[index] */
        if (v->requires_grad) tape_append(OP_SELECT, v, t, NULL, (double)index);
        return v;
    } else if (t->rank == 2 && dim == 0) {
        int cols = t->shape[1];
        int shape[] = {cols};
        /* Row selection: share data with parent */
        Tensor* r = calloc(1, sizeof(Tensor));
        r->data = t->data + index * cols;
        r->shape = malloc(sizeof(int));
        r->shape[0] = cols;
        r->rank = 1;
        r->numel = cols;
        r->requires_grad = t->requires_grad;
        r->tape_idx = -1;
        r->grad = NULL;
        if (r->requires_grad) tape_append(OP_SELECT, r, t, NULL, (double)index);
        return r;
    }
    return make_scalar(t->data[index], t->requires_grad);
}

TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) {
    /* Stack scalars into a 1D vector */
    double* data = malloc(count * sizeof(double));
    for (int i = 0; i < count; i++) data[i] = ((Tensor*)tensors[i])->data[0];
    int shape[] = {count};
    Tensor* r = make_tensor(data, shape, 1, 0);
    free(data);
    return r;
}

TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) {
    return tensor_stack(tensors, count, dim); /* simplified */
}

/* ================================================================
   Autograd — backward pass
   ================================================================ */



void tensor_backward(TensorHandle h) {
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) return;

    /* Initialize loss gradient to 1.0 */
    ensure_grad(loss);
    loss->grad[0] = 1.0;
    

    /* Walk tape in reverse */

    for (int i = loss->tape_idx; i >= 0; i--) {
        TapeEntry* e = &tape[i];
        Tensor* r = e->result;
        if (!r->grad) continue;

        Tensor* a = e->arg1;
        Tensor* b = e->arg2;

        switch (e->op) {
        case OP_CONST: break; /* leaf — grad already accumulated */

        case OP_ADD:
            if (a) { ensure_grad(a); ensure_grad(r);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[a->numel == 1 ? 0 : j];
                /* broadcast: if a is scalar and r is multi-dim, sum the grads */
                if (a->numel == 1 && r->numel > 1) { double s=0; for(int j=1;j<r->numel;j++) s+=r->grad[j]; a->grad[0]+=s; }
            }
            if (b) { ensure_grad(b); ensure_grad(r);
                for (int j = 0; j < b->numel; j++) b->grad[j] += r->grad[b->numel == 1 ? 0 : j];
                if (b->numel == 1 && r->numel > 1) { double s=0; for(int j=1;j<r->numel;j++) s+=r->grad[j]; b->grad[0]+=s; }
            }
            break;

        case OP_SUB:
            if (a) { ensure_grad(a); ensure_grad(r);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[a->numel == 1 ? 0 : j];
                if (a->numel == 1 && r->numel > 1) { double s=0; for(int j=1;j<r->numel;j++) s+=r->grad[j]; a->grad[0]+=s; }
            }
            if (b) { ensure_grad(b); ensure_grad(r);
                for (int j = 0; j < b->numel; j++) b->grad[j] -= r->grad[b->numel == 1 ? 0 : j];
                if (b->numel == 1 && r->numel > 1) { double s=0; for(int j=1;j<r->numel;j++) s-=r->grad[j]; b->grad[0]+=s; }
            }
            break;

        case OP_MUL:
            if (a) { ensure_grad(a); ensure_grad(r);
                if (a->numel == 1 && b->numel > 1) {
                    double s = 0; for (int j = 0; j < b->numel; j++) s += r->grad[j] * b->data[j];
                    a->grad[0] += s;
                } else {
                    for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j] * b->data[b->numel==1?0:j];
                }
            }
            if (b) { ensure_grad(b); ensure_grad(r);
                if (b->numel == 1 && a->numel > 1) {
                    double s = 0; for (int j = 0; j < a->numel; j++) s += r->grad[j] * a->data[j];
                    b->grad[0] += s;
                } else {
                    for (int j = 0; j < b->numel; j++) b->grad[j] += r->grad[j] * a->data[a->numel==1?0:j];
                }
            }
            break;

        case OP_DIV:
            if (a) {
                ensure_grad(a); ensure_grad(r);
                if (a->numel == 1) {
                    a->grad[0] += r->grad[0] / b->data[0];
                } else {
                    /* a is multi-dim, b is scalar: d(a/b)/da = 1/b for each element */
                    for (int j = 0; j < a->numel; j++)
                        a->grad[j] += r->grad[j] / b->data[b->numel == 1 ? 0 : j];
                }
            }
            if (b) {
                ensure_grad(b); ensure_grad(r);
                if (b->numel == 1 && a->numel > 1) {
                    /* d(a/b)/db = -sum(a_i * grad_i) / b^2 */
                    double s = 0;
                    for (int j = 0; j < a->numel; j++)
                        s += r->grad[j] * a->data[j];
                    b->grad[0] -= s / (b->data[0] * b->data[0]);
                } else {
                    b->grad[0] -= r->grad[0] * a->data[0] / (b->data[0] * b->data[0]);
                }
            }
            break;

        case OP_NEG:
            if (a) { ensure_grad(a); a->grad[0] -= r->grad[0]; }
            break;

        case OP_ABS:
            if (a) { ensure_grad(a); a->grad[0] += r->grad[0] * (a->data[0] >= 0 ? 1.0 : -1.0); }
            break;

        case OP_EXP:
            if (a) { ensure_grad(a); a->grad[0] += r->grad[0] * r->data[0]; }
            break;

        case OP_LOG:
            if (a) { ensure_grad(a); a->grad[0] += r->grad[0] / a->data[0]; }
            break;

        case OP_SQRT:
            if (a) { ensure_grad(a); a->grad[0] += r->grad[0] / (2.0 * r->data[0]); }
            break;

        case OP_POW:
            if (a) {
                ensure_grad(a); ensure_grad(r);
                if (a->numel > 1 && b->numel == 1) {
                    /* a is [n], b is scalar: d(a^b)/da_i = b * a_i^(b-1) * grad_i */
                    double bv = b->data[0];
                    for (int j = 0; j < a->numel; j++) {
                        double av = fmax(a->data[j], 1e-20); /* prevent pow(0, neg) */
                        a->grad[j] += r->grad[j] * bv * pow(av, bv - 1.0);
                    }
                } else {
                    a->grad[0] += r->grad[0] * b->data[0] * pow(fmax(a->data[0], 1e-20), b->data[0] - 1.0);
                }
            }
            if (b) {
                ensure_grad(b); ensure_grad(r);
                if (b->numel == 1 && a->numel > 1) {
                    /* d(a^b)/db = sum(a_i^b * log(a_i) * grad_i) */
                    double s = 0;
                    for (int j = 0; j < a->numel; j++) {
                        double av = fmax(a->data[j], 1e-20);
                        s += r->grad[j] * r->data[j] * log(av);
                    }
                    b->grad[0] += s;
                } else {
                    b->grad[0] += r->grad[0] * r->data[0] * log(fmax(a->data[0], 1e-20));
                }
            }
            break;

        case OP_SIGMOID: {
            double s = r->data[0];
            if (a) { ensure_grad(a); a->grad[0] += r->grad[0] * s * (1.0 - s); }
            break;
        }

        case OP_TANH: {
            double t = r->data[0];
            if (a) { ensure_grad(a); a->grad[0] += r->grad[0] * (1.0 - t * t); }
            break;
        }

        case OP_ADD_SCALAR:
            if (a) {
                ensure_grad(a); ensure_grad(r);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j];
            }
            break;

        case OP_MUL_SCALAR:
            if (a) {
                ensure_grad(a); ensure_grad(r);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j] * e->scalar_arg;
            }
            break;

        case OP_CLAMP_MIN: {
            /* Gradient passes through where input > min, zero where clamped */
            double min_val = e->scalar_arg;
            if (a) {
                ensure_grad(a);
                ensure_grad(r);
                for (int j = 0; j < a->numel; j++)
                    a->grad[j] += (a->data[j] > min_val) ? r->grad[j] : 0.0;
            }
            break;
        }

        case OP_SELECT: {
            /* Select: grad of parent[index] += grad of result */
            int sel_idx = (int)e->scalar_arg;
            if (a) {
                ensure_grad(a);
                ensure_grad(r);
                if (r->numel == 1) {
                    /* Scalar select from vector */
                    a->grad[sel_idx] += r->grad[0];
                } else {
                    /* Row select from matrix */
                    int cols = r->numel;
                    for (int j = 0; j < cols; j++)
                        a->grad[sel_idx * cols + j] += r->grad[j];
                }
            }
            break;
        }

        case OP_RESHAPE:
            /* Reshape is a view — gradient passes through unchanged */
            if (a) {
                ensure_grad(a);
                ensure_grad(r);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j];
            }
            break;

        case OP_STACK:
            /* Distribute gradient from stacked tensor back to constituent scalars */
            if (e->inputs) {
                for (int j = 0; j < e->input_count; j++) {
                    Tensor* inp = e->inputs[j];
                    if (inp->requires_grad) {
                        ensure_grad(inp);
                        ensure_grad(r);
                        inp->grad[0] += r->grad[j];
                    }
                }
            }
            break;

        case OP_SUM:
            if (a) {
                ensure_grad(a);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[0];
            }
            break;

        case OP_MEAN:
            if (a) {
                ensure_grad(a);
                double scale = 1.0 / a->numel;
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[0] * scale;
            }
            break;

        case OP_DOT:
            /* d(dot(a,b))/da = b, d(dot(a,b))/db = a (element-wise) */
            if (a && a->numel > 1) {
                ensure_grad(a);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[0] * b->data[j];
            } else if (a) {
                ensure_grad(a);
                a->grad[0] += r->grad[0] * b->data[0];
            }
            if (b && b->numel > 1) {
                ensure_grad(b);
                for (int j = 0; j < b->numel; j++) b->grad[j] += r->grad[0] * a->data[j];
            } else if (b) {
                ensure_grad(b);
                b->grad[0] += r->grad[0] * a->data[0];
            }
            break;

        case OP_MV: {
            /* d(Ax)/dA = grad ⊗ x, d(Ax)/dx = A^T @ grad */
            int m = a->shape[0], n_mv = a->shape[1];
            ensure_grad(r);
            if (a->requires_grad) {
                ensure_grad(a);
                for (int ii = 0; ii < m; ii++)
                    for (int jj = 0; jj < n_mv; jj++)
                        a->grad[ii*n_mv+jj] += r->grad[ii] * b->data[jj];
            }
            if (b->requires_grad) {
                ensure_grad(b);
                for (int jj = 0; jj < n_mv; jj++) {
                    double s = 0;
                    for (int ii = 0; ii < m; ii++) s += a->data[ii*n_mv+jj] * r->grad[ii];
                    b->grad[jj] += s;
                }
            }
            break;
        }

        case OP_OUTER: {
            /* d(outer(a,b))/da[i] = sum_j(grad[i,j] * b[j]) */
            /* d(outer(a,b))/db[j] = sum_i(grad[i,j] * a[i]) */
            int m_out = a->numel, n_out = b->numel;
            if (a->requires_grad) {
                ensure_grad(a);
                ensure_grad(r);
                for (int ii = 0; ii < m_out; ii++) {
                    double s = 0;
                    for (int jj = 0; jj < n_out; jj++) s += r->grad[ii*n_out+jj] * b->data[jj];
                    a->grad[ii] += s;
                }
            }
            if (b->requires_grad) {
                ensure_grad(b);
                ensure_grad(r);
                for (int jj = 0; jj < n_out; jj++) {
                    double s = 0;
                    for (int ii = 0; ii < m_out; ii++) s += r->grad[ii*n_out+jj] * a->data[ii];
                    b->grad[jj] += s;
                }
            }
            break;
        }

        case OP_SOFTMAX: {
            /* Softmax backward: d/dx_i = sum_j(grad_j * sm_j * (delta_ij - sm_i)) */
            if (a) {
                ensure_grad(a);
                ensure_grad(r);
                int n_sm = r->numel;
                for (int ii = 0; ii < n_sm; ii++) {
                    double s = 0;
                    for (int jj = 0; jj < n_sm; jj++) {
                        double delta = (ii == jj) ? 1.0 : 0.0;
                        s += r->grad[jj] * r->data[jj] * (delta - r->data[ii]);
                    }
                    a->grad[ii] += s;
                }
            }
            break;
        }

        case OP_BCE_WITH_LOGITS: {
            /* d/dp_i = (1/n) * (sigmoid(p_i) - y_i) */
            if (a) {
                ensure_grad(a);
                int n_bce = a->numel;
                for (int j = 0; j < n_bce; j++) {
                    double sig = 1.0 / (1.0 + exp(-a->data[j]));
                    a->grad[j] += r->grad[0] * (sig - b->data[j]) / n_bce;
                }
            }
            break;
        }

        case OP_COSINE_SIM: {
            /* Cosine similarity backward: a=[n,w] matrix, b=[1,w] key (unsqueezed) */
            if (a && a->rank == 2 && b && b->rank == 2) {
                int n_cs = a->shape[0], w_cs = a->shape[1];
                double* brow = b->data;
                double bnorm2 = 0;
                for (int j = 0; j < w_cs; j++) bnorm2 += brow[j] * brow[j];
                double bnorm = sqrt(bnorm2) + 1e-8;

                ensure_grad(a); ensure_grad(r);
                for (int ii = 0; ii < n_cs; ii++) {
                    double anorm2 = 0;
                    for (int j = 0; j < w_cs; j++) anorm2 += a->data[ii*w_cs+j] * a->data[ii*w_cs+j];
                    double anorm = sqrt(anorm2) + 1e-8;
                    double cos_val = r->data[ii];
                    double g = r->grad[ii];
                    for (int j = 0; j < w_cs; j++) {
                        a->grad[ii*w_cs+j] += g * (brow[j] / (anorm * bnorm) - cos_val * a->data[ii*w_cs+j] / (anorm2 + 1e-10));
                    }
                }

                if (b->requires_grad) {
                    ensure_grad(b);
                    for (int ii = 0; ii < n_cs; ii++) {
                        double anorm2 = 0;
                        for (int j = 0; j < w_cs; j++) anorm2 += a->data[ii*w_cs+j] * a->data[ii*w_cs+j];
                        double anorm = sqrt(anorm2) + 1e-8;
                        double cos_val = r->data[ii];
                        double g = r->grad[ii];
                        for (int j = 0; j < w_cs; j++) {
                            b->grad[j] += g * (a->data[ii*w_cs+j] / (anorm * bnorm) - cos_val * brow[j] / (bnorm2 + 1e-10));
                        }
                    }
                }
            }
            break;
        }

        case OP_CONV1D_CIRC: {
            /* Circular convolution backward */
            int n_cv = a->numel, k_cv = b->numel, pad_cv = k_cv / 2;
            ensure_grad(r);
            if (a->requires_grad) {
                ensure_grad(a);
                for (int ii = 0; ii < n_cv; ii++) {
                    for (int j = 0; j < k_cv; j++) {
                        int idx = (ii - pad_cv + j + n_cv) % n_cv;
                        a->grad[idx] += r->grad[ii] * b->data[k_cv - 1 - j];
                    }
                }
            }
            if (b->requires_grad) {
                ensure_grad(b);
                for (int ii = 0; ii < n_cv; ii++) {
                    for (int j = 0; j < k_cv; j++) {
                        int idx = (ii - pad_cv + j + n_cv) % n_cv;
                        b->grad[k_cv - 1 - j] += r->grad[ii] * a->data[idx];
                    }
                }
            }
            break;
        }

        default: break; /* unimplemented backward */
        }
    }
}

TensorHandle tensor_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (!t->grad) return NULL;
    return make_scalar(t->grad[0], 0);
}

void tensor_zero_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
}

int tensor_requires_grad(TensorHandle h) {
    return ((Tensor*)h)->requires_grad;
}

TensorHandle tensor_detach(TensorHandle h) {
    return tensor_clone(h);
}

TensorHandle tensor_with_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    Tensor* r = make_scalar(t->data[0], 1);
    tape_append(OP_CONST, r, NULL, NULL, 0);
    return r;
}

void tensor_set_requires_grad(TensorHandle h, int rg) {
    Tensor* t = (Tensor*)h;
    t->requires_grad = rg;
    if (rg && t->tape_idx < 0) {
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

void tensor_no_grad_begin(void) { /* no-op for tape backend */ }
void tensor_no_grad_end(void) { }

/* ================================================================
   Device (CPU only)
   ================================================================ */

TensorHandle tensor_to_device(TensorHandle t, const char* device) { return t; }
const char* tensor_device(TensorHandle t) { return "cpu"; }

/* ================================================================
   LSTM
   ================================================================ */

void tensor_lstm_cell(
    TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh,
    TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c)
{
    /* Stub — not used in current Idris code (uses tensor_lstm_gates_pair) */
    *out_h = tensor_clone(hx);
    *out_c = tensor_clone(cx);
}

void tensor_lstm_gates(TensorHandle combined_h, TensorHandle prev_cell_h, int o,
                       TensorHandle* out_h, TensorHandle* out_c)
{
    Tensor* combined = (Tensor*)combined_h;
    Tensor* prev_cell = (Tensor*)prev_cell_h;
    double* out_hidden = calloc(o, sizeof(double));
    double* out_cell = calloc(o, sizeof(double));

    for (int j = 0; j < o; j++) {
        double ig = 1.0 / (1.0 + exp(-combined->data[j]));         /* input gate */
        double fg = 1.0 / (1.0 + exp(-combined->data[o+j]));       /* forget gate */
        double gg = tanh(combined->data[2*o+j]);                     /* cell gate */
        double og = 1.0 / (1.0 + exp(-combined->data[3*o+j]));     /* output gate */
        out_cell[j] = fg * prev_cell->data[j] + ig * gg;
        out_hidden[j] = og * tanh(out_cell[j]);
    }

    int shape[] = {o};
    *out_h = make_tensor(out_hidden, shape, 1, 0);
    *out_c = make_tensor(out_cell, shape, 1, 0);
    free(out_hidden);
    free(out_cell);
}

TensorPair* tensor_lstm_gates_pair(TensorHandle combined, TensorHandle prev_cell, int o) {
    TensorPair* p = malloc(sizeof(TensorPair));
    tensor_lstm_gates(combined, prev_cell, o, &p->first, &p->second);
    return p;
}

TensorHandle tensor_pair_first(TensorPair* p) { return p->first; }
TensorHandle tensor_pair_second(TensorPair* p) { return p->second; }
void tensor_pair_free(TensorPair* p) { free(p); }

/* ================================================================
   Parameter Registry
   ================================================================ */

typedef struct {
    char name[256];
    Tensor* tensor;
} ParamEntry;

#define MAX_PARAMS 65536
static ParamEntry param_registry[MAX_PARAMS];
static int param_count_val = 0;

void param_register(const char* name, TensorHandle h) {
    Tensor* t = (Tensor*)h;
    /* Replace if exists */
    for (int i = 0; i < param_count_val; i++) {
        if (strcmp(param_registry[i].name, name) == 0) {
            param_registry[i].tensor = t;
            return;
        }
    }
    if (param_count_val < MAX_PARAMS) {
        strncpy(param_registry[param_count_val].name, name, 255);
        param_registry[param_count_val].tensor = t;
        param_count_val++;
    }
}

void param_clear(void) { param_count_val = 0; }
int param_count(void) { return param_count_val; }
const char* param_name(int idx) { return param_registry[idx].name; }

double param_grad_item(int idx) {
    Tensor* t = param_registry[idx].tensor;
    if (!t->grad) return 0.0;
    return t->grad[0];
}

double param_grad_item_and_zero(int idx) {
    Tensor* t = param_registry[idx].tensor;
    if (!t->grad) return 0.0;
    double v = t->grad[0];
    t->grad[0] = 0.0;
    return v;
}

TensorHandle param_tensor(int idx) { return param_registry[idx].tensor; }

void param_zero_all_grads(void) {
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
    }
}

void param_subtract_delta(int idx, double delta) {
    Tensor* t = param_registry[idx].tensor;
    t->data[0] -= delta;
}

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    Tensor* t = (Tensor*)h;
    for (int i = 0; i < t->numel; i++) t->data[i] -= val;
    return h;
}

/* ================================================================
   Convenience functions
   ================================================================ */

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    int shape[] = {n};
    return tensor_create(data, shape, 1, requires_grad);
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    int shape[] = {rows, cols};
    return tensor_create(data, shape, 2, requires_grad);
}

double* tensor_alloc_doubles(int n) { return calloc(n, sizeof(double)); }
double tensor_read_double(double* buf, int idx) { return buf[idx]; }
void tensor_write_double(double* buf, int idx, double val) { buf[idx] = val; }

TensorHandle* tensor_ptr_array_alloc(int n) {
    return calloc(n, sizeof(TensorHandle));
}

void tensor_ptr_array_set(TensorHandle* arr, int idx, TensorHandle t) {
    arr[idx] = t;
}

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    double* data = malloc(count * sizeof(double));
    int rg = 0;
    Tensor** inputs = malloc(count * sizeof(Tensor*));
    for (int i = 0; i < count; i++) {
        Tensor* t = (Tensor*)arr[i];
        data[i] = t->data[0];
        inputs[i] = t;
        if (t->requires_grad) rg = 1;
    }
    free(arr);
    int shape[] = {count};
    Tensor* r = make_tensor(data, shape, 1, rg);
    free(data);
    if (rg) {
        int idx = tape_append(OP_STACK, r, NULL, NULL, 0);
        tape[idx].inputs = inputs;
        tape[idx].input_count = count;
    } else {
        free(inputs);
    }
    return r;
}

TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    return tensor_stack_from_array(arr, count, dim);
}

/* ================================================================
   Tensor-level parameter creation
   ================================================================ */

TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    /* Param tensors use regular malloc — persist across arena resets */
    int numel = rows * cols;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    t->shape = malloc(2 * sizeof(int));
    t->shape[0] = rows; t->shape[1] = cols;
    t->rank = 2; t->numel = numel;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

TensorHandle tensor_create_param_1d(int n, double* data) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(n * sizeof(double));
    memcpy(t->data, data, n * sizeof(double));
    t->shape = malloc(sizeof(int));
    t->shape[0] = n;
    t->rank = 1; t->numel = n;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    Tensor* t = (Tensor*)h;
    int cols = t->shape[1];
    /* View shares data with parent (data pointer into parent's array).
       NOT requires_grad — the parent handles grad tracking. */
    Tensor* v = calloc(1, sizeof(Tensor));
    v->data = &t->data[row * cols + col];
    v->shape = NULL;
    v->rank = 0;
    v->numel = 1;
    v->requires_grad = 0;
    v->tape_idx = -1;
    v->grad = NULL;
    return v;
}

TensorHandle tensor_view_1d(TensorHandle h, int idx) {
    Tensor* t = (Tensor*)h;
    Tensor* v = calloc(1, sizeof(Tensor));
    v->data = &t->data[idx];
    v->shape = NULL;
    v->rank = 0;
    v->numel = 1;
    v->requires_grad = 0;
    v->tape_idx = -1;
    v->grad = NULL;
    return v;
}

double tensor_item_2d(TensorHandle h, int row, int col) {
    Tensor* t = (Tensor*)h;
    return t->data[row * t->shape[1] + col];
}

double tensor_item_1d(TensorHandle h, int idx) {
    return ((Tensor*)h)->data[idx];
}

/* ================================================================
   Native Optimizer
   ================================================================ */

typedef struct {
    double lr;
    int type; /* 0=SGD, 1=RMSprop, 2=Adam */
    double alpha, eps, weight_decay, momentum;
    double beta1, beta2;
    double* v;  /* second moment (RMSprop/Adam) */
    double* m;  /* first moment (Adam) / momentum buffer (RMSprop) */
    int t;      /* step count */
    int allocated;
} Optimizer;

static void optimizer_ensure_buffers(Optimizer* opt) {
    if (opt->allocated) return;
    int n = param_count_val;
    opt->v = calloc(n, sizeof(double));
    opt->m = calloc(n, sizeof(double));
    opt->allocated = 1;
}

OptimizerHandle optimizer_create_sgd(double lr) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr;
    opt->type = 0;
    return opt;
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                          double weight_decay, double momentum) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 1;
    opt->alpha = alpha; opt->eps = eps;
    opt->weight_decay = weight_decay; opt->momentum = momentum;
    return opt;
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 2;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    return opt;
}

void optimizer_free(OptimizerHandle h) {
    Optimizer* opt = (Optimizer*)h;
    free(opt->v); free(opt->m); free(opt);
}

void optimizer_zero_grad(OptimizerHandle h) {
    (void)h;
    param_zero_all_grads();
}

void optimizer_step(OptimizerHandle h) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    opt->t++;


    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;

        for (int j = 0; j < t->numel; j++) {
            double g = t->grad[j];

            switch (opt->type) {
            case 0: /* SGD */
                t->data[j] -= opt->lr * g;
                break;

            case 1: { /* RMSprop */
                int idx = i; /* simplified: 1 slot per param (scalar) */
                opt->v[idx] = opt->alpha * opt->v[idx] + (1.0 - opt->alpha) * g * g;
                double delta = opt->lr * g / (sqrt(opt->v[idx]) + opt->eps);
                if (opt->momentum > 0) {
                    opt->m[idx] = opt->momentum * opt->m[idx] + delta;
                    t->data[j] -= opt->m[idx];
                } else {
                    t->data[j] -= delta;
                }
                break;
            }

            case 2: { /* Adam */
                int idx = i;
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                t->data[j] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
                break;
            }
            }
        }
    }

    /* Reset tape and re-register ONLY the param tensors (from param_registry).
       Ephemeral tensors (select results, intermediates) are not re-registered.
       They will be recreated in the next forward pass. */
    tape_reset();
    for (int j = 0; j < param_count_val; j++) {
        Tensor* t = param_registry[j].tensor;
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

void optimizer_clip_grad_value(double max_val) {
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) {
            if (t->grad[j] > max_val) t->grad[j] = max_val;
            if (t->grad[j] < -max_val) t->grad[j] = -max_val;
        }
    }
}

double optimizer_clip_grad_norm(double max_norm) {
    double total = 0;
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) total += t->grad[j] * t->grad[j];
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count_val; i++) {
            Tensor* t = param_registry[i].tensor;
            if (!t->grad) continue;
            for (int j = 0; j < t->numel; j++) t->grad[j] *= scale;
        }
    }
    return norm;
}

/* ================================================================
   System
   ================================================================ */

int backend_supports_tensor_params(void) { return 0; }

int get_rss_mb(void) {
#ifdef __APPLE__
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (int)(usage.ru_maxrss / (1024 * 1024));
#else
    return 0;
#endif
}

int get_current_rss_mb(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return (int)(info.resident_size / (1024 * 1024));
#endif
    return get_rss_mb();
}

/* ================================================================
   Debug
   ================================================================ */

void tensor_print(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (t->rank == 0) {
        printf("%.6f\n", t->data[0]);
    } else {
        printf("[");
        for (int i = 0; i < t->numel; i++) {
            if (i > 0) printf(", ");
            printf("%.6f", t->data[i]);
        }
        printf("]\n");
    }
}

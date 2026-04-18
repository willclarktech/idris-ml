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
            size_t cap = bytes > ARENA_INIT_SIZE ? bytes : ARENA_INIT_SIZE;
            ArenaChunk* c = arena_new_chunk(cap);
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
    OP_LSTM_GATES_CELL,  /* cell output — shares LstmGatesMeta with OP_LSTM_GATES */
    OP_STACK,     /* stack of scalar tensors into 1D */
    OP_RESHAPE,   /* reshape (view) — grad passes through unchanged */
    OP_SELECT,    /* select element from vector — grad goes to parent[index] */
    OP_VECMAT,    /* [n] x [n,m] -> [m] vector-matrix multiply */
    OP_CAT,       /* concatenate two 1D tensors: [a] ++ [b] -> [a+b] */
    OP_NARROW,    /* view into a slice of a 1D tensor */
    OP_NTM_READ_HEAD_READ, /* read output — shares NtmReadHeadMeta with OP_NTM_READ_HEAD */
    OP_LOG_SOFTMAX_2D, /* row-wise log-softmax on [m,n] */
    OP_MM,            /* [m,n] x [n,k] -> [m,k] matrix-matrix multiply */
    OP_TRANSPOSE_2D,  /* [m,n] -> [n,m] transpose */
    OP_SOFTMAX_2D,    /* row-wise softmax on [m,n] */
    OP_MASKED_FILL,   /* fill masked positions with a value */
    OP_LAYER_NORM_2D, /* row-wise layer normalization on [m,n] */
    OP_BMM,           /* batched matrix multiply: [B,m,n] x [n,k] -> [B,m,k] */
    OP_BMM_3X3,       /* batched matmul: [B,m,n] x [B,n,k] -> [B,m,k] */
    OP_SOFTMAX_3D,    /* row-wise softmax on [B,m,n] along last dim */
    OP_TRANSPOSE_LAST2, /* [B,m,n] -> [B,n,m] */
    OP_GELU,          /* GELU activation (tanh approximation) */
    OP_EMBEDDING,     /* row gather from weight matrix */
    OP_BATCH_NORM,    /* per-channel normalization */
    OP_DROPOUT,       /* inverted dropout with stored mask */
    OP_AVG_POOL1D,    /* [C,L] -> [C,oL] mean pooling */
    OP_AVG_POOL2D,    /* [C,H,W] -> [C,oH,oW] mean pooling */
    OP_CONV1D,        /* [inC,L] * [outC,inC,kL] + [outC] -> [outC,oL] */
    OP_MAX_POOL1D,    /* [C,L] -> [C,oL] with max indices */
    OP_CONV2D,        /* [inC,H,W] * [outC,inC,kH,kW] + [outC] -> [outC,oH,oW] */
    OP_MAX_POOL2D,    /* [C,H,W] -> [C,oH,oW] with max indices */
};

typedef struct {
    int op;
    Tensor* result;     /* non-owning pointer to the result tensor */
    Tensor* arg1;       /* non-owning: first input */
    Tensor* arg2;       /* non-owning: second input (NULL for unary) */
    double scalar_arg;  /* for add_scalar, mul_scalar */
    Tensor** inputs;    /* for OP_STACK: array of constituent scalar tensors */
    int input_count;    /* number of inputs for stack */
    void* op_meta;      /* op-specific metadata for backward (arena-allocated) */
} TapeEntry;

/* Op metadata structs for fused backward */
typedef struct { int m, n; double* x_vals; } MvMeta;
typedef struct { int n; double* out_vals; } SoftmaxMeta;
typedef struct {
    int o;
    double* iG; double* fG; double* gG; double* oG;  /* activated gate values */
    double* new_cell;
} LstmGatesMeta;
typedef struct {
    int n, w;
    double key_norm;
    double* row_norms;
    double* dots;
} CosSimMeta;

typedef struct {
    int n, w, k;
    Tensor* memory;
    Tensor* prev_weights;
    Tensor* key;
    Tensor* beta;
    Tensor* g;
    Tensor* gamma;
    Tensor* shift;
    double beta_v, g_v, gamma_v;
    double key_norm;
    double* row_norms;     /* [n] */
    double* raw_cos;       /* [n] pre-beta cosine similarity */
    double* content_weights; /* [n] softmax output */
    double* interp;        /* [n] interpolated weights */
    double* shifted;       /* [n] after circular conv */
    double* shifted_clamped; /* [n] after clamp */
    double* powered;       /* [n] clamped^gamma */
    double pow_sum;
    double* focused;       /* [n] final normalized weights */
} NtmReadHeadMeta;

typedef struct {
    int n, w;
    Tensor* add_vector;
} NtmInterpWriteMeta;

typedef struct {
    Tensor* gamma;     /* scale parameter [n] */
    Tensor* bias;      /* shift parameter [n] */
    double* x_hat;     /* normalized values [m*n] */
    double* rstd;      /* reciprocal std devs [m] */
    int m, n;
} LayerNormMeta;

typedef struct {
    int n, embedDim;
    int* indices;  /* [n] integer indices, heap-allocated */
} EmbeddingMeta;

typedef struct {
    Tensor* gamma;
    Tensor* beta;
    double* x_hat;     /* normalized values [C * spatial], heap-allocated */
    double* rstd;      /* reciprocal std devs [C], heap-allocated */
    int C, spatial;
} BatchNormMeta;

typedef struct {
    double* mask;   /* [numel] binary mask (0 or 1/(1-p)), heap-allocated */
    int numel;
} DropoutMeta;

typedef struct {
    int C, L, kL, stride, oL;
} AvgPool1DMeta;

typedef struct {
    int C, H, W, kH, kW, strH, strW, oH, oW;
} AvgPool2DMeta;

typedef struct {
    int inC, outC, L, kL, pad, stride, oL;
} Conv1DMeta;

typedef struct {
    int C, L, kL, stride, oL;
    int* max_indices;
} MaxPool1DMeta;

typedef struct {
    int inC, outC, H, W, kH, kW, padH, padW, strH, strW, oH, oW;
} Conv2DMeta;

typedef struct {
    int C, H, W, kH, kW, strH, strW, oH, oW;
    int* max_indices;  /* [C * oH * oW] index into flat input per-channel */
} MaxPool2DMeta;

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
    memset(&tape[idx], 0, sizeof(TapeEntry));
    tape[idx].op = op;
    tape[idx].result = result;
    tape[idx].arg1 = arg1;
    tape[idx].arg2 = arg2;
    tape[idx].scalar_arg = scalar_arg;
    result->tape_idx = idx;
    return idx;
}

static void tape_reset(void) {
    /* Free heap-allocated resources before clearing tape */
    for (int i = 0; i < tape_size; i++) {
        /* Free OP_STACK inputs arrays */
        if (tape[i].op == OP_STACK && tape[i].inputs) {
            free(tape[i].inputs);
            tape[i].inputs = NULL;
        }
        /* Free OP_LAYER_NORM_2D heap arrays */
        if (tape[i].op == OP_LAYER_NORM_2D && tape[i].op_meta) {
            LayerNormMeta* meta = (LayerNormMeta*)tape[i].op_meta;
            free(meta->x_hat);
            free(meta->rstd);
            meta->x_hat = NULL;
            meta->rstd = NULL;
        }
        /* Free OP_EMBEDDING indices */
        if (tape[i].op == OP_EMBEDDING && tape[i].op_meta) {
            EmbeddingMeta* meta = (EmbeddingMeta*)tape[i].op_meta;
            free(meta->indices);
            meta->indices = NULL;
        }
        /* Free OP_BATCH_NORM arrays */
        if (tape[i].op == OP_BATCH_NORM && tape[i].op_meta) {
            BatchNormMeta* meta = (BatchNormMeta*)tape[i].op_meta;
            free(meta->x_hat);
            free(meta->rstd);
            meta->x_hat = NULL;
            meta->rstd = NULL;
        }
        /* Free OP_DROPOUT mask */
        if (tape[i].op == OP_DROPOUT && tape[i].op_meta) {
            DropoutMeta* meta = (DropoutMeta*)tape[i].op_meta;
            free(meta->mask);
            meta->mask = NULL;
        }
        /* Free OP_MAX_POOL1D max indices */
        if (tape[i].op == OP_MAX_POOL1D && tape[i].op_meta) {
            MaxPool1DMeta* meta = (MaxPool1DMeta*)tape[i].op_meta;
            free(meta->max_indices);
            meta->max_indices = NULL;
        }
        /* Free OP_MAX_POOL2D max indices */
        if (tape[i].op == OP_MAX_POOL2D && tape[i].op_meta) {
            MaxPool2DMeta* meta = (MaxPool2DMeta*)tape[i].op_meta;
            free(meta->max_indices);
            meta->max_indices = NULL;
        }
        /* Free grad arrays on non-persistent (arena) tensors.
           These are heap-allocated by ensure_grad during backward. */
        Tensor* r = tape[i].result;
        if (r && !r->persistent && r->grad) {
            free(r->grad);
            r->grad = NULL;
        }
    }
    tape_size = 0;
    arena_reset();
}

/* ================================================================
   Lifecycle
   ================================================================ */

static int persistent_scalar_count = 0;

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    /* Always heap-allocate: these are returned to Idris and may be cached
       in Variables across epochs (surviving arena_reset). The per-epoch leak
       from training data tensors (~15KB/epoch) is acceptable. */
    persistent_scalar_count++;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(sizeof(double));
    t->data[0] = value;
    t->rank = 0; t->numel = 1;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->persistent = 1;
    if (requires_grad) tape_append(OP_CONST, t, NULL, NULL, 0);
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

/* Unary ops: support both scalar (rank 0) and multi-element tensors */
static double fn_neg(double x) { return -x; }
static double fn_abs(double x) { return fabs(x); }
static double fn_exp_d(double x) { return exp(x); }
static double fn_log_d(double x) { return log(x); }
static double fn_sqrt_d(double x) { return sqrt(x); }

static TensorHandle unop_elementwise(TensorHandle ha, int op, double (*fn)(double)) {
    Tensor* a = (Tensor*)ha;
    if (a->numel == 1) {
        /* Scalar path (backward rules use [0] indexing) */
        Tensor* r = make_scalar(fn(a->data[0]), a->requires_grad);
        if (a->requires_grad) tape_append(op, r, a, NULL, 0);
        return r;
    }
    /* Multi-element: apply fn element-wise, preserve shape */
    double* data = malloc(a->numel * sizeof(double));
    for (int i = 0; i < a->numel; i++) data[i] = fn(a->data[i]);
    Tensor* r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    free(data);
    if (a->requires_grad) tape_append(op, r, a, NULL, 0);
    return r;
}

TensorHandle tensor_neg(TensorHandle a) { return unop_elementwise(a, OP_NEG, fn_neg); }
TensorHandle tensor_abs(TensorHandle a) { return unop_elementwise(a, OP_ABS, fn_abs); }
TensorHandle tensor_exp(TensorHandle a) { return unop_elementwise(a, OP_EXP, fn_exp_d); }
TensorHandle tensor_log(TensorHandle a) { return unop_elementwise(a, OP_LOG, fn_log_d); }
TensorHandle tensor_sqrt(TensorHandle a) { return unop_elementwise(a, OP_SQRT, fn_sqrt_d); }

static double fn_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double fn_tanh_d(double x) { return tanh(x); }

TensorHandle tensor_sigmoid(TensorHandle a) { return unop_elementwise(a, OP_SIGMOID, fn_sigmoid); }

TensorHandle tensor_tanh(TensorHandle a) { return unop_elementwise(a, OP_TANH, fn_tanh_d); }

/* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static double fn_gelu_d(double x) {
    double c = 0.7978845608028654;  /* sqrt(2/pi) */
    double inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}
TensorHandle tensor_gelu(TensorHandle a) { return unop_elementwise(a, OP_GELU, fn_gelu_d); }

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
    if (r->requires_grad) {
        int idx = tape_append(OP_MV, r, mat, vec, 0);
        /* Save input values for backward (input may be arena-allocated, freed before backward) */
        MvMeta* meta = arena_alloc(sizeof(MvMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        memcpy(meta->x_vals, vec->data, n * sizeof(double));
        tape[idx].op_meta = meta;
    }
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
        /* [n] x [n,m] = [m] — row vector × matrix */
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
        if (r->requires_grad) tape_append(OP_VECMAT, r, a, b, 0);
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

/* Matrix-matrix multiply: [m,n] x [n,k] -> [m,k] */
TensorHandle tensor_mm(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    int m = a->shape[0], n = a->shape[1], k = b->shape[1];
    int rg = a->requires_grad || b->requires_grad;
    double* data = calloc(m * k, sizeof(double));

#ifdef __APPLE__
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, k, n, 1.0, a->data, n, b->data, k, 0.0, data, k);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            double s = 0;
            for (int p = 0; p < n; p++) s += a->data[i*n+p] * b->data[p*k+j];
            data[i*k+j] = s;
        }
#endif

    int shape[] = {m, k};
    Tensor* r = make_tensor(data, shape, 2, rg);
    free(data);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return r;
}

/* Batched matrix-matrix multiply: [B,m,n] x [n,k] -> [B,m,k]
   Weight matrix b is shared across all batch elements. */
TensorHandle tensor_bmm(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    int B = a->shape[0], m = a->shape[1], n = a->shape[2], k = b->shape[1];
    int rg = a->requires_grad || b->requires_grad;
    double* data = calloc(B * m * k, sizeof(double));

    for (int bi = 0; bi < B; bi++) {
#ifdef __APPLE__
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, n, 1.0,
                    a->data + bi * m * n, n,
                    b->data, k,
                    0.0, data + bi * m * k, k);
#else
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++) {
                double s = 0;
                for (int p = 0; p < n; p++)
                    s += a->data[bi*m*n + i*n+p] * b->data[p*k+j];
                data[bi*m*k + i*k+j] = s;
            }
#endif
    }

    int shape[] = {B, m, k};
    Tensor* r = make_tensor(data, shape, 3, rg);
    free(data);
    if (rg) tape_append(OP_BMM, r, a, b, 0);
    return r;
}

TensorHandle tensor_bmm_3x3(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    int B = a->shape[0], m = a->shape[1], n = a->shape[2], k = b->shape[2];
    int rg = a->requires_grad || b->requires_grad;
    double* data = calloc(B * m * k, sizeof(double));

    for (int bi = 0; bi < B; bi++) {
#ifdef __APPLE__
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, n, 1.0,
                    a->data + bi * m * n, n,
                    b->data + bi * n * k, k,
                    0.0, data + bi * m * k, k);
#else
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++) {
                double s = 0;
                for (int p = 0; p < n; p++)
                    s += a->data[bi*m*n + i*n+p] * b->data[bi*n*k + p*k+j];
                data[bi*m*k + i*k+j] = s;
            }
#endif
    }

    int shape[] = {B, m, k};
    Tensor* r = make_tensor(data, shape, 3, rg);
    free(data);
    if (rg) tape_append(OP_BMM_3X3, r, a, b, 0);
    return r;
}

TensorHandle tensor_softmax_3d(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int B = t->shape[0], m = t->shape[1], n = t->shape[2];
    int total_rows = B * m;
    double* data = malloc(t->numel * sizeof(double));

    for (int i = 0; i < total_rows; i++) {
        double max_val = t->data[i*n];
        for (int j = 1; j < n; j++)
            if (t->data[i*n+j] > max_val) max_val = t->data[i*n+j];
        double sum = 0;
        for (int j = 0; j < n; j++) {
            data[i*n+j] = exp(t->data[i*n+j] - max_val);
            sum += data[i*n+j];
        }
        for (int j = 0; j < n; j++)
            data[i*n+j] /= sum;
    }

    int shape[] = {B, m, n};
    Tensor* r = make_tensor(data, shape, 3, t->requires_grad);
    free(data);
    if (t->requires_grad) tape_append(OP_SOFTMAX_3D, r, t, NULL, 0);
    return r;
}

TensorHandle tensor_transpose_last2(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int B = t->shape[0], m = t->shape[1], n = t->shape[2];
    double* data = malloc(t->numel * sizeof(double));

    for (int bi = 0; bi < B; bi++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                data[bi*n*m + j*m + i] = t->data[bi*m*n + i*n + j];

    int shape[] = {B, n, m};
    Tensor* r = make_tensor(data, shape, 3, t->requires_grad);
    free(data);
    if (t->requires_grad) tape_append(OP_TRANSPOSE_LAST2, r, t, NULL, 0);
    return r;
}

TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
    int shape[] = {d0, d1, d2};
    return tensor_reshape(h, shape, 3);
}

TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
    Tensor* mask = (Tensor*)hmask;
    int mn = mask->numel;
    double* data = malloc(B * mn * sizeof(double));
    for (int bi = 0; bi < B; bi++)
        memcpy(data + bi * mn, mask->data, mn * sizeof(double));
    int shape[] = {B, mask->shape[0], mask->shape[1]};
    Tensor* r = make_tensor(data, shape, 3, 0);
    free(data);
    return r;
}

/* Stack B tensors of shape [m, n] into [B, m, n].
   All tensors must have the same shape. No gradient tracking (data tensors). */
TensorHandle tensor_batch(TensorHandle* handles, int count) {
    Tensor* first = (Tensor*)handles[0];
    int elem_size = first->numel;
    int total = count * elem_size;
    double* data = malloc(total * sizeof(double));
    for (int i = 0; i < count; i++) {
        Tensor* t = (Tensor*)handles[i];
        memcpy(data + i * elem_size, t->data, elem_size * sizeof(double));
    }
    /* Build shape: [count, first->shape[0], ..., first->shape[rank-1]] */
    int rank = first->rank + 1;
    int* shape = malloc(rank * sizeof(int));
    shape[0] = count;
    for (int i = 0; i < first->rank; i++) shape[i+1] = first->shape[i];
    Tensor* r = make_tensor(data, shape, rank, 0);
    free(data);
    free(shape);
    return r;
}

/* Split [B, ...] tensor into B tensors of shape [...].
   Returns array of B tensor handles (caller must free array). */
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
    Tensor* t = (Tensor*)h;
    int B = t->shape[0];
    *out_count = B;
    int elem_size = t->numel / B;
    int inner_rank = t->rank - 1;
    TensorHandle* handles = malloc(B * sizeof(TensorHandle));
    for (int i = 0; i < B; i++) {
        Tensor* r = arena_alloc(sizeof(Tensor));
        memset(r, 0, sizeof(Tensor));
        r->data = t->data + i * elem_size;  /* view into parent */
        r->shape = arena_alloc(inner_rank * sizeof(int));
        for (int j = 0; j < inner_rank; j++) r->shape[j] = t->shape[j+1];
        r->rank = inner_rank;
        r->numel = elem_size;
        r->requires_grad = t->requires_grad;
        r->tape_idx = -1;
        r->persistent = 0;
        handles[i] = (TensorHandle)r;
    }
    return handles;
}

/* Transpose 2D tensor: [m,n] -> [n,m] */
TensorHandle tensor_transpose_2d(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int m = t->shape[0], n = t->shape[1];
    double* data = malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            data[j*m+i] = t->data[i*n+j];
    int shape[] = {n, m};
    Tensor* r = make_tensor(data, shape, 2, t->requires_grad);
    free(data);
    if (r->requires_grad) tape_append(OP_TRANSPOSE_2D, r, t, NULL, 0);
    return r;
}

/* Row-wise softmax on 2D tensor: [m,n] -> [m,n], each row sums to 1 */
TensorHandle tensor_softmax_2d(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int m = t->shape[0], n = t->shape[1];
    double* data = malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        double max_val = t->data[i*n];
        for (int j = 1; j < n; j++)
            if (t->data[i*n+j] > max_val) max_val = t->data[i*n+j];
        double sum_exp = 0;
        for (int j = 0; j < n; j++) {
            data[i*n+j] = exp(t->data[i*n+j] - max_val);
            sum_exp += data[i*n+j];
        }
        for (int j = 0; j < n; j++) data[i*n+j] /= sum_exp;
    }
    int shape[] = {m, n};
    Tensor* r = make_tensor(data, shape, 2, t->requires_grad);
    free(data);
    if (r->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, NULL, 0);
    return r;
}

/* Row-wise log-softmax on 2D tensor: [m,n] -> [m,n] */
TensorHandle tensor_log_softmax_2d(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    int m = t->shape[0], n = t->shape[1];
    double* data = malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        double max_val = t->data[i*n];
        for (int j = 1; j < n; j++)
            if (t->data[i*n+j] > max_val) max_val = t->data[i*n+j];
        double sum_exp = 0;
        for (int j = 0; j < n; j++) sum_exp += exp(t->data[i*n+j] - max_val);
        double log_sum = log(sum_exp) + max_val;
        for (int j = 0; j < n; j++) data[i*n+j] = t->data[i*n+j] - log_sum;
    }
    int shape[] = {m, n};
    Tensor* r = make_tensor(data, shape, 2, t->requires_grad);
    free(data);
    if (r->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, NULL, 0);
    return r;
}

/* Element-wise multiply for multi-element tensors (same shape) */
TensorHandle tensor_mul_elementwise(TensorHandle ha, TensorHandle hb) {
    return tensor_mul(ha, hb);  /* tensor_mul already handles multi-element via binop_elementwise */
}

/* Sum all elements of a tensor (not just scalar) */
TensorHandle tensor_sum_all(TensorHandle h) {
    return tensor_sum(h);  /* tensor_sum already sums all elements */
}

/* Masked fill: replace positions where mask[i]=1 with value */
TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle hmask, double value) {
    Tensor* t = (Tensor*)h;
    Tensor* mask = (Tensor*)hmask;
    double* data = malloc(t->numel * sizeof(double));
    for (int i = 0; i < t->numel; i++)
        data[i] = (mask->data[i] != 0.0) ? value : t->data[i];
    Tensor* r = make_tensor(data, t->shape, t->rank, t->requires_grad);
    free(data);
    if (r->requires_grad) tape_append(OP_MASKED_FILL, r, t, mask, 0);
    return r;
}

/* Create upper-triangular causal mask [n,n]: 1.0 above diagonal, 0.0 on/below */
TensorHandle tensor_causal_mask(int n) {
    double* data = calloc(n * n, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            data[i*n+j] = 1.0;
    int shape[] = {n, n};
    Tensor* r = make_tensor(data, shape, 2, 0);  /* no grad needed for mask */
    free(data);
    return r;
}

/* Row-wise layer normalization on 2D tensor: y[i,j] = gamma[j] * x_hat[i,j] + beta[j]
   where x_hat[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) */
TensorHandle tensor_layer_norm_2d(TensorHandle h, TensorHandle hgamma,
                                   TensorHandle hbias, double eps) {
    Tensor* t = (Tensor*)h;
    Tensor* gamma = (Tensor*)hgamma;
    Tensor* bias = (Tensor*)hbias;
    int m = t->shape[0], n = t->shape[1];
    double* data = malloc(m * n * sizeof(double));
    double* x_hat = malloc(m * n * sizeof(double));
    double* rstd = malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        /* Compute mean */
        double mean = 0;
        for (int j = 0; j < n; j++) mean += t->data[i*n+j];
        mean /= n;
        /* Compute variance */
        double var = 0;
        for (int j = 0; j < n; j++) {
            double d = t->data[i*n+j] - mean;
            var += d * d;
        }
        var /= n;
        double inv_std = 1.0 / sqrt(var + eps);
        rstd[i] = inv_std;
        /* Normalize and apply affine */
        for (int j = 0; j < n; j++) {
            x_hat[i*n+j] = (t->data[i*n+j] - mean) * inv_std;
            data[i*n+j] = gamma->data[j] * x_hat[i*n+j] + bias->data[j];
        }
    }
    int shape[] = {m, n};
    int rg = t->requires_grad || gamma->requires_grad || bias->requires_grad;
    Tensor* r = make_tensor(data, shape, 2, rg);
    free(data);
    if (rg) {
        /* Store x_hat and rstd in persistent (heap) arrays since arena resets */
        LayerNormMeta* meta = arena_alloc(sizeof(LayerNormMeta));
        meta->gamma = gamma;
        meta->bias = bias;
        meta->x_hat = x_hat;  /* heap-allocated, freed in tape_reset */
        meta->rstd = rstd;    /* heap-allocated, freed in tape_reset */
        meta->m = m;
        meta->n = n;
        int idx = tape_append(OP_LAYER_NORM_2D, r, t, NULL, 0);
        tape[idx].op_meta = meta;
    } else {
        free(x_hat);
        free(rstd);
    }
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
    if (r->requires_grad) {
        int idx = tape_append(OP_SOFTMAX, r, t, NULL, 0);
        SoftmaxMeta* meta = arena_alloc(sizeof(SoftmaxMeta));
        meta->n = n;
        meta->out_vals = r->data;  /* r persists in arena — safe to reference */
        tape[idx].op_meta = meta;
    }
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

/* ================================================================
   Embedding: row gather from weight matrix
   weight [vocabSize, embedDim], indices [n] -> output [n * embedDim]
   ================================================================ */

TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    Tensor* weight = (Tensor*)hweight;
    Tensor* indices = (Tensor*)hindices;
    int out_numel = n * embedDim;

    double* out = calloc(out_numel, sizeof(double));
    int* idx_copy = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        int idx = (int)indices->data[i];
        idx_copy[i] = idx;
        memcpy(out + i * embedDim, weight->data + idx * embedDim, embedDim * sizeof(double));
    }

    int out_shape[] = {out_numel};
    Tensor* r = make_tensor(out, out_shape, 1, weight->requires_grad);
    free(out);

    if (r->requires_grad) {
        int tape_idx = tape_append(OP_EMBEDDING, r, weight, NULL, 0);
        EmbeddingMeta* meta = arena_alloc(sizeof(EmbeddingMeta));
        meta->n = n;
        meta->embedDim = embedDim;
        meta->indices = idx_copy;
        tape[tape_idx].op_meta = meta;
    } else {
        free(idx_copy);
    }
    return r;
}

/* ================================================================
   Batch Normalization: per-channel, across spatial dims
   Input treated as [C, spatial]. Normalizes each channel independently.
   ================================================================ */

TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               TensorHandle hrunning_mean, TensorHandle hrunning_var,
                               int C, int spatial, int training,
                               double momentum, double eps) {
    Tensor* input = (Tensor*)hinput;
    Tensor* gamma = (Tensor*)hgamma;
    Tensor* beta = (Tensor*)hbeta;
    Tensor* running_mean = (Tensor*)hrunning_mean;
    Tensor* running_var = (Tensor*)hrunning_var;
    int n = C * spatial;

    double* out = calloc(n, sizeof(double));
    double* x_hat = malloc(n * sizeof(double));
    double* rstd = malloc(C * sizeof(double));

    for (int c = 0; c < C; c++) {
        double mean, var;
        if (training) {
            /* Compute mean and var from input for this channel */
            mean = 0;
            for (int j = 0; j < spatial; j++) mean += input->data[c * spatial + j];
            mean /= spatial;
            var = 0;
            for (int j = 0; j < spatial; j++) {
                double d = input->data[c * spatial + j] - mean;
                var += d * d;
            }
            var /= spatial;
            /* Update running stats (in-place, no grad) */
            running_mean->data[c] = (1.0 - momentum) * running_mean->data[c] + momentum * mean;
            running_var->data[c] = (1.0 - momentum) * running_var->data[c] + momentum * var;
        } else {
            mean = running_mean->data[c];
            var = running_var->data[c];
        }

        double rs = 1.0 / sqrt(var + eps);
        rstd[c] = rs;
        for (int j = 0; j < spatial; j++) {
            int idx = c * spatial + j;
            x_hat[idx] = (input->data[idx] - mean) * rs;
            out[idx] = gamma->data[c] * x_hat[idx] + beta->data[c];
        }
    }

    int out_shape[1] = {n};
    int rg = input->requires_grad || gamma->requires_grad || beta->requires_grad;
    Tensor* r = make_tensor(out, out_shape, 1, rg);
    free(out);

    if (r->requires_grad) {
        int idx = tape_append(OP_BATCH_NORM, r, input, NULL, 0);
        BatchNormMeta* meta = arena_alloc(sizeof(BatchNormMeta));
        meta->gamma = gamma;
        meta->beta = beta;
        meta->x_hat = x_hat;
        meta->rstd = rstd;
        meta->C = C;
        meta->spatial = spatial;
        tape[idx].op_meta = meta;
    } else {
        free(x_hat);
        free(rstd);
    }
    return r;
}

/* ================================================================
   Dropout: inverted dropout with mask
   ================================================================ */

TensorHandle tensor_dropout(TensorHandle hinput, double p, int training, unsigned int seed) {
    Tensor* input = (Tensor*)hinput;
    int n = input->numel;

    if (!training || p <= 0.0) return hinput;  /* eval mode or p=0: identity */

    double scale = 1.0 / (1.0 - p);
    double* out = arena_alloc(n * sizeof(double));
    double* mask = malloc(n * sizeof(double));  /* heap: survives for backward */

    for (int i = 0; i < n; i++) {
        /* Simple LCG per-element (fast, deterministic per seed) */
        seed = seed * 1103515245u + 12345u;
        double r = (double)((seed >> 16) & 0x7fff) / 32767.0;
        if (r < p) {
            mask[i] = 0.0;
            out[i] = 0.0;
        } else {
            mask[i] = scale;
            out[i] = input->data[i] * scale;
        }
    }

    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = out;
    r->shape = input->shape;  /* share shape (same dims) */
    r->rank = input->rank;
    r->numel = n;
    r->requires_grad = input->requires_grad;
    r->tape_idx = -1;
    r->persistent = 0;

    if (r->requires_grad) {
        int idx = tape_append(OP_DROPOUT, r, input, NULL, 0);
        DropoutMeta* meta = arena_alloc(sizeof(DropoutMeta));
        meta->mask = mask;
        meta->numel = n;
        tape[idx].op_meta = meta;
    } else {
        free(mask);
    }
    return r;
}

/* ================================================================
   Gather / Scatter
   ================================================================ */

TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    Tensor* input = (Tensor*)hinput;
    Tensor* index = (Tensor*)hindex;
    double* out = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        int idx = (int)index->data[i];
        out[i] = input->data[idx];
    }
    int shape[] = {n};
    Tensor* r = make_tensor(out, shape, 1, input->requires_grad);
    free(out);
    /* No tape entry needed — gather is a read-only index operation.
       For training, embedding lookups use scatter_add in backward. */
    return r;
}

TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
    Tensor* index = (Tensor*)hindex;
    Tensor* src = (Tensor*)hsrc;
    double* out = calloc(out_size, sizeof(double));
    for (int i = 0; i < src->numel; i++) {
        int idx = (int)index->data[i];
        if (idx >= 0 && idx < out_size)
            out[idx] += src->data[i];
    }
    int shape[] = {out_size};
    Tensor* r = make_tensor(out, shape, 1, src->requires_grad);
    free(out);
    return r;
}

/* ================================================================
   Average Pooling
   ================================================================ */

TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], L = input->shape[1];
    int oL = (L - kL) / stride + 1;
    double scale = 1.0 / kL;
    double* out = calloc(C * oL, sizeof(double));
    for (int c = 0; c < C; c++)
        for (int ol = 0; ol < oL; ol++) {
            double s = 0;
            for (int kl = 0; kl < kL; kl++) s += input->data[c*L + ol*stride + kl];
            out[c*oL + ol] = s * scale;
        }
    int out_shape[] = {C, oL};
    Tensor* r = make_tensor(out, out_shape, 2, input->requires_grad);
    free(out);
    if (r->requires_grad) {
        int idx = tape_append(OP_AVG_POOL1D, r, input, NULL, 0);
        AvgPool1DMeta* meta = arena_alloc(sizeof(AvgPool1DMeta));
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        tape[idx].op_meta = meta;
    }
    return r;
}

TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    double scale = 1.0 / (kH * kW);
    double* out = calloc(C * oH * oW, sizeof(double));
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) {
                double s = 0;
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                        s += input->data[c*H*W + (oh*strideH+kh)*W + ow*strideW+kw];
                out[c*oH*oW + oh*oW + ow] = s * scale;
            }
    int out_shape[] = {C, oH, oW};
    Tensor* r = make_tensor(out, out_shape, 3, input->requires_grad);
    free(out);
    if (r->requires_grad) {
        int idx = tape_append(OP_AVG_POOL2D, r, input, NULL, 0);
        AvgPool2DMeta* meta = arena_alloc(sizeof(AvgPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW; meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].op_meta = meta;
    }
    return r;
}

/* ================================================================
   Conv1D: input [inC, L], kernel [outC, inC, kL], bias [outC] or NULL
   Output: [outC, oL]
   ================================================================ */

TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int pad, int stride) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;

    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[0], kL = kernel->shape[2];
    int oL = (L + 2*pad - kL) / stride + 1;

    double* out = calloc(outC * oL, sizeof(double));
    for (int oc = 0; oc < outC; oc++) {
        for (int ol = 0; ol < oL; ol++) {
            double val = bias ? bias->data[oc] : 0.0;
            for (int ic = 0; ic < inC; ic++)
                for (int kl = 0; kl < kL; kl++) {
                    int il = ol * stride - pad + kl;
                    if (il >= 0 && il < L)
                        val += input->data[ic*L + il] * kernel->data[oc*inC*kL + ic*kL + kl];
                }
            out[oc*oL + ol] = val;
        }
    }

    int out_shape[] = {outC, oL};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor(out, out_shape, 2, rg);
    free(out);

    if (r->requires_grad) {
        int idx = tape_append(OP_CONV1D, r, input, kernel, 0);
        Conv1DMeta* meta = arena_alloc(sizeof(Conv1DMeta));
        meta->inC = inC; meta->outC = outC; meta->L = L;
        meta->kL = kL; meta->pad = pad; meta->stride = stride; meta->oL = oL;
        tape[idx].op_meta = meta;
        tape[idx].inputs = (Tensor**)bias;
    }
    return r;
}

TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], L = input->shape[1];
    int oL = (L - kL) / stride + 1;

    double* out = calloc(C * oL, sizeof(double));
    int* max_idx = malloc(C * oL * sizeof(int));

    for (int c = 0; c < C; c++)
        for (int ol = 0; ol < oL; ol++) {
            double best = -1e30;
            int best_idx = 0;
            for (int kl = 0; kl < kL; kl++) {
                int il = ol * stride + kl;
                int flat = c*L + il;
                if (input->data[flat] > best) { best = input->data[flat]; best_idx = flat; }
            }
            out[c*oL + ol] = best;
            max_idx[c*oL + ol] = best_idx;
        }

    int out_shape[] = {C, oL};
    Tensor* r = make_tensor(out, out_shape, 2, input->requires_grad);
    free(out);

    if (r->requires_grad) {
        int idx = tape_append(OP_MAX_POOL1D, r, input, NULL, 0);
        MaxPool1DMeta* meta = arena_alloc(sizeof(MaxPool1DMeta));
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        meta->max_indices = max_idx;
        tape[idx].op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    int numel = d0 * d1 * d2;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);
    t->shape = malloc(3 * sizeof(int));
    t->shape[0] = d0; t->shape[1] = d1; t->shape[2] = d2;
    t->rank = 3; t->numel = numel;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

/* ================================================================
   Conv2D: input [inC, H, W], kernel [outC, inC, kH, kW], bias [outC] or NULL
   Output: [outC, oH, oW]
   ================================================================ */

TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;

    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[0], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int out_numel = outC * oH * oW;

    double* out = calloc(out_numel, sizeof(double));

    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double val = bias ? bias->data[oc] : 0.0;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * strideH - padH + kh;
                            int iw = ow * strideW - padW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                val += input->data[ic*H*W + ih*W + iw]
                                     * kernel->data[oc*inC*kH*kW + ic*kH*kW + kh*kW + kw];
                            }
                        }
                    }
                }
                out[oc*oH*oW + oh*oW + ow] = val;
            }
        }
    }

    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor(out, out_shape, 3, rg);
    free(out);

    if (r->requires_grad) {
        int idx = tape_append(OP_CONV2D, r, input, kernel, 0);
        Conv2DMeta* meta = arena_alloc(sizeof(Conv2DMeta));
        meta->inC = inC; meta->outC = outC;
        meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].op_meta = meta;
        /* Store bias pointer in scalar_arg slot (cast) for backward */
        tape[idx].inputs = (Tensor**)bias;  /* reuse inputs field for bias ptr */
    }
    return r;
}

/* ================================================================
   MaxPool2D: input [C, H, W] -> [C, oH, oW]
   ================================================================ */

TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    int out_numel = C * oH * oW;

    double* out = calloc(out_numel, sizeof(double));
    int* max_idx = malloc(out_numel * sizeof(int));

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double best = -1e30;
                int best_idx = 0;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * strideH + kh;
                        int iw = ow * strideW + kw;
                        int flat = c*H*W + ih*W + iw;
                        if (input->data[flat] > best) {
                            best = input->data[flat];
                            best_idx = flat;
                        }
                    }
                }
                int out_idx = c*oH*oW + oh*oW + ow;
                out[out_idx] = best;
                max_idx[out_idx] = best_idx;
            }
        }
    }

    int out_shape[] = {C, oH, oW};
    Tensor* r = make_tensor(out, out_shape, 3, input->requires_grad);
    free(out);

    if (r->requires_grad) {
        int idx = tape_append(OP_MAX_POOL2D, r, input, NULL, 0);
        MaxPool2DMeta* meta = arena_alloc(sizeof(MaxPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;  /* heap-allocated, freed in tape_reset */
        tape[idx].op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

/* Backward helper: propagate d_focused through the NTM read head addressing chain */
static void ntm_read_head_backward_chain(NtmReadHeadMeta* m, double* d_focused) {
    int n = m->n, w = m->w, k = m->k;
    int pad = k / 2;
    double S = m->pow_sum + 1e-10;

    /* Step 1: d_focused → d_powered (division normalization backward) */
    double dot_fg = 0;
    for (int i = 0; i < n; i++) dot_fg += d_focused[i] * m->focused[i];
    double* d_powered = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        d_powered[i] = (d_focused[i] - dot_fg) / S;

    /* Step 2: d_powered → d_clamped + d_gamma (power backward) */
    double* d_clamped = calloc(n, sizeof(double));
    double d_gamma = 0;
    for (int i = 0; i < n; i++) {
        double c = m->shifted_clamped[i];
        d_clamped[i] = d_powered[i] * m->gamma_v * pow(c, m->gamma_v - 1.0);
        if (c > 1e-10)  /* avoid log(~0) */
            d_gamma += d_powered[i] * m->powered[i] * log(c);
    }

    /* Step 3: d_clamped → d_shifted (clamp backward) */
    double* d_shifted = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        d_shifted[i] = (m->shifted[i] > 1e-10) ? d_clamped[i] : 0.0;

    /* Step 4: d_shifted → d_interp + d_shift (circular conv backward) */
    double* d_interp = calloc(n, sizeof(double));
    double* d_shift_k = calloc(k, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) {
            int idx = (i - pad + j + n) % n;
            d_interp[idx] += d_shifted[i] * m->shift->data[k - 1 - j];
            d_shift_k[k - 1 - j] += d_shifted[i] * m->interp[idx];
        }

    /* Step 5: d_interp → d_content + d_prev + d_g (interpolation backward) */
    double* d_content = calloc(n, sizeof(double));
    double d_g = 0;
    for (int i = 0; i < n; i++) {
        d_content[i] = d_interp[i] * m->g_v;
        d_g += d_interp[i] * (m->content_weights[i] - m->prev_weights->data[i]);
    }
    if (m->prev_weights->requires_grad) {
        ensure_grad(m->prev_weights);
        for (int i = 0; i < n; i++)
            m->prev_weights->grad[i] += d_interp[i] * (1.0 - m->g_v);
    }

    /* Step 6: d_content → d_scaled (softmax backward) */
    double dot_dc = 0;
    for (int i = 0; i < n; i++) dot_dc += d_content[i] * m->content_weights[i];
    double* d_scaled = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        d_scaled[i] = m->content_weights[i] * (d_content[i] - dot_dc);

    /* Step 7: d_scaled → d_beta + d_raw_cos */
    double d_beta = 0;
    for (int i = 0; i < n; i++) d_beta += d_scaled[i] * m->raw_cos[i];
    /* d_raw_cos[i] = d_scaled[i] * beta */
    /* Step 8: d_raw_cos → d_memory + d_key (cosine similarity backward) */
    double kn = m->key_norm, kn2 = kn * kn;
    if (m->memory->requires_grad) {
        ensure_grad(m->memory);
        for (int i = 0; i < n; i++) {
            double mn = m->row_norms[i], mn2 = mn * mn;
            double d_rc = d_scaled[i] * m->beta_v;  /* d_raw_cos[i] */
            for (int j = 0; j < w; j++)
                m->memory->grad[i*w+j] += d_rc * (m->key->data[j] / (mn * kn)
                    - m->raw_cos[i] * m->memory->data[i*w+j] / mn2);
        }
    }
    if (m->key->requires_grad) {
        ensure_grad(m->key);
        for (int i = 0; i < n; i++) {
            double mn = m->row_norms[i];
            double d_rc = d_scaled[i] * m->beta_v;
            for (int j = 0; j < w; j++)
                m->key->grad[j] += d_rc * (m->memory->data[i*w+j] / (mn * kn)
                    - m->raw_cos[i] * m->key->data[j] / kn2);
        }
    }

    /* Scalar gradients */
    if (m->beta->requires_grad) { ensure_grad(m->beta); m->beta->grad[0] += d_beta; }
    if (m->g->requires_grad) { ensure_grad(m->g); m->g->grad[0] += d_g; }
    if (m->gamma->requires_grad) { ensure_grad(m->gamma); m->gamma->grad[0] += d_gamma; }
    if (m->shift->requires_grad) {
        ensure_grad(m->shift);
        for (int j = 0; j < k; j++) m->shift->grad[j] += d_shift_k[j];
    }

    free(d_powered); free(d_clamped); free(d_shifted);
    free(d_interp); free(d_shift_k); free(d_content);
    free(d_scaled);
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
    int kk = shift->numel, pad = kk / 2;
    double beta_v = beta->data[0], g_v = g_t->data[0], gamma_v = gamma->data[0];
    int rg = memory->requires_grad || prev_w->requires_grad || key->requires_grad
           || beta->requires_grad || g_t->requires_grad || gamma->requires_grad
           || shift->requires_grad;

    NtmReadHeadMeta* meta = NULL;
    if (rg) {
        meta = arena_alloc(sizeof(NtmReadHeadMeta));
        meta->n = n; meta->w = w; meta->k = kk;
        meta->memory = memory; meta->prev_weights = prev_w;
        meta->key = key; meta->beta = beta; meta->g = g_t;
        meta->gamma = gamma; meta->shift = shift;
        meta->beta_v = beta_v; meta->g_v = g_v; meta->gamma_v = gamma_v;
        meta->row_norms = arena_alloc(n * sizeof(double));
        meta->raw_cos = arena_alloc(n * sizeof(double));
        meta->content_weights = arena_alloc(n * sizeof(double));
        meta->interp = arena_alloc(n * sizeof(double));
        meta->shifted = arena_alloc(n * sizeof(double));
        meta->shifted_clamped = arena_alloc(n * sizeof(double));
        meta->powered = arena_alloc(n * sizeof(double));
        meta->focused = arena_alloc(n * sizeof(double));
    }

    /* 1. Content addressing */
    double* cos_sim = calloc(n, sizeof(double));
    double key_norm = 0;
    for (int j = 0; j < w; j++) key_norm += key->data[j] * key->data[j];
    key_norm = sqrt(key_norm) + 1e-8;
    if (meta) meta->key_norm = key_norm;
    for (int i = 0; i < n; i++) {
        double dot = 0, row_norm = 0;
        for (int j = 0; j < w; j++) {
            dot += memory->data[i*w+j] * key->data[j];
            row_norm += memory->data[i*w+j] * memory->data[i*w+j];
        }
        row_norm = sqrt(row_norm) + 1e-8;
        double rc = dot / (row_norm * key_norm);
        if (meta) { meta->row_norms[i] = row_norm; meta->raw_cos[i] = rc; }
        cos_sim[i] = beta_v * rc;
    }
    /* softmax */
    double max_cs = cos_sim[0];
    for (int i = 1; i < n; i++) if (cos_sim[i] > max_cs) max_cs = cos_sim[i];
    double sum_exp = 0;
    for (int i = 0; i < n; i++) { cos_sim[i] = exp(cos_sim[i] - max_cs); sum_exp += cos_sim[i]; }
    for (int i = 0; i < n; i++) cos_sim[i] /= sum_exp;
    if (meta) memcpy(meta->content_weights, cos_sim, n * sizeof(double));

    /* 2. Interpolation */
    double* interp = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        interp[i] = g_v * cos_sim[i] + (1.0 - g_v) * prev_w->data[i];
    if (meta) memcpy(meta->interp, interp, n * sizeof(double));

    /* 3. Circular shift */
    double* shifted = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < kk; j++) {
            int idx = (i - pad + j + n) % n;
            s += interp[idx] * shift->data[kk - 1 - j];
        }
        shifted[i] = s;
    }
    if (meta) memcpy(meta->shifted, shifted, n * sizeof(double));

    /* 4. Sharpening */
    double* focused = calloc(n, sizeof(double));
    double pow_sum = 0;
    for (int i = 0; i < n; i++) {
        double clamped = fmax(shifted[i], 1e-10);
        if (meta) meta->shifted_clamped[i] = clamped;
        focused[i] = pow(clamped, gamma_v);
        if (meta) meta->powered[i] = focused[i];
        pow_sum += focused[i];
    }
    if (meta) meta->pow_sum = pow_sum;
    for (int i = 0; i < n; i++) focused[i] /= (pow_sum + 1e-10);
    if (meta) memcpy(meta->focused, focused, n * sizeof(double));

    /* 5. Read */
    double* read_out = calloc(w, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < w; j++)
            read_out[j] += focused[i] * memory->data[i*w+j];

    int w_shape[] = {n};
    int r_shape[] = {w};
    TensorPair* pair = arena_alloc(sizeof(TensorPair));
    pair->first = make_tensor(focused, w_shape, 1, rg);
    pair->second = make_tensor(read_out, r_shape, 1, rg);

    if (rg) {
        int idx_w = tape_append(OP_NTM_READ_HEAD, (Tensor*)pair->first, NULL, NULL, 0);
        tape[idx_w].op_meta = meta;
        int idx_r = tape_append(OP_NTM_READ_HEAD_READ, (Tensor*)pair->second, NULL, NULL, 0);
        tape[idx_r].op_meta = meta;
    }

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
    int rg = memory->requires_grad || weights->requires_grad || add_vec->requires_grad;
    double* data = malloc(n * w * sizeof(double));
    memcpy(data, memory->data, n * w * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < w; j++)
            data[i*w+j] += weights->data[i] * add_vec->data[j];
    int shape[] = {n, w};
    Tensor* r = make_tensor(data, shape, 2, rg);
    free(data);
    if (rg) {
        NtmInterpWriteMeta* meta = arena_alloc(sizeof(NtmInterpWriteMeta));
        meta->n = n; meta->w = w; meta->add_vector = add_vec;
        int idx = tape_append(OP_NTM_INTERP_WRITE, r, memory, weights, 0);
        tape[idx].op_meta = meta;
    }
    return r;
}

/* ================================================================
   Shape manipulation
   ================================================================ */

TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    Tensor* t = (Tensor*)h;
    /* Create a new tensor with different shape but shared data (arena-allocated) */
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = t->data;  /* shared */
    r->shape = arena_alloc(rank * sizeof(int));
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
    /* Scalar: select(scalar, 0, 0) is identity — return the tensor directly
       to preserve tape connectivity (the scalar already has a tape entry). */
    if (t->rank == 0) return h;
    if (t->rank == 1) {
        Tensor* v = arena_alloc(sizeof(Tensor));
        memset(v, 0, sizeof(Tensor));
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
        /* Row selection: share data with parent */
        Tensor* r = arena_alloc(sizeof(Tensor));
        memset(r, 0, sizeof(Tensor));
        r->data = t->data + index * cols;
        r->shape = arena_alloc(sizeof(int));
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
    return tensor_stack(tensors, count, dim); /* simplified: scalar-only */
}

/* Concatenate two 1D tensors: [a] ++ [b] -> [a+b] */
TensorHandle tensor_cat2(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    int na = a->numel, nb = b->numel, total = na + nb;
    int rg = a->requires_grad || b->requires_grad;
    double* data = arena_alloc(total * sizeof(double));
    memcpy(data, a->data, na * sizeof(double));
    memcpy(data + na, b->data, nb * sizeof(double));
    int* shape = arena_alloc(sizeof(int));
    shape[0] = total;
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = data; r->shape = shape; r->rank = 1;
    r->numel = total; r->requires_grad = rg;
    r->tape_idx = -1;
    /* OP_CAT stores a as arg1, b as arg2. scalar_arg = split point (na) */
    if (rg) tape_append(OP_CAT, r, a, b, (double)na);
    return r;
}

/* View into a slice of a 1D tensor: t[start..start+len) */
TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    Tensor* t = (Tensor*)h;
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = t->data + start;
    r->shape = arena_alloc(sizeof(int));
    r->shape[0] = len;
    r->rank = 1; r->numel = len;
    r->requires_grad = t->requires_grad;
    r->tape_idx = -1;
    /* OP_NARROW: scatter gradient back to parent at offset */
    if (r->requires_grad) tape_append(OP_NARROW, r, t, NULL, (double)start);
    return r;
}

/* ================================================================
   Autograd — backward pass
   ================================================================ */

/* Forward declarations for profiling */
#include <sys/time.h>
static double _wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
static double prof_forward_ms = 0, prof_backward_ms = 0, prof_optimizer_ms = 0;
static int prof_forward_ops = 0, prof_backward_ops = 0, prof_epochs = 0;



void tensor_backward(TensorHandle h) {
    double t0 = _wall_ms();
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) return;

    /* Initialize loss gradient to 1.0 */
    ensure_grad(loss);
    loss->grad[0] = 1.0;

    int processed = 0, skipped = 0;

    /* Walk tape in reverse */

    for (int i = loss->tape_idx; i >= 0; i--) {
        TapeEntry* e = &tape[i];
        Tensor* r = e->result;
        if (!r->grad) { skipped++; continue; }
        processed++;

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
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) a->grad[j] -= r->grad[j]; }
            break;

        case OP_ABS:
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j] * (a->data[j] >= 0 ? 1.0 : -1.0); }
            break;

        case OP_EXP:
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j] * r->data[j]; }
            break;

        case OP_LOG:
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j] / a->data[j]; }
            break;

        case OP_SQRT:
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j] / (2.0 * r->data[j]); }
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
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) { double s = r->data[j]; a->grad[j] += r->grad[j] * s * (1.0 - s); } }
            break;
        }

        case OP_TANH: {
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) { double t = r->data[j]; a->grad[j] += r->grad[j] * (1.0 - t * t); } }
            break;
        }

        case OP_GELU: {
            /* d_gelu/dx = 0.5*(1+tanh(inner)) + 0.5*x*(1-tanh^2)*c*(1+3*0.044715*x^2) */
            if (a) {
                ensure_grad(a);
                double c = 0.7978845608028654;
                for (int j = 0; j < a->numel; j++) {
                    double x = a->data[j];
                    double inner = c * (x + 0.044715 * x * x * x);
                    double t = tanh(inner);
                    double dtdx = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x);
                    a->grad[j] += r->grad[j] * (0.5 * (1.0 + t) + 0.5 * x * dtdx);
                }
            }
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

        case OP_VECMAT: {
            /* r[j] = sum_i a[i] * b[i*m+j], where a=[n], b=[n,m], r=[m] */
            int n_vm = a->numel;
            int m_vm = r->numel;
            ensure_grad(r);
            if (a) {
                /* d_a[i] = sum_j r_grad[j] * b[i*m+j] */
                ensure_grad(a);
                for (int i = 0; i < n_vm; i++) {
                    double s = 0;
                    for (int j = 0; j < m_vm; j++) s += r->grad[j] * b->data[i*m_vm+j];
                    a->grad[i] += s;
                }
            }
            if (b) {
                /* d_b[i*m+j] = r_grad[j] * a[i] */
                ensure_grad(b);
                for (int i = 0; i < n_vm; i++)
                    for (int j = 0; j < m_vm; j++)
                        b->grad[i*m_vm+j] += r->grad[j] * a->data[i];
            }
            break;
        }

        case OP_CAT: {
            /* r = cat(a, b), split at scalar_arg */
            int split = (int)e->scalar_arg;
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j];
            }
            if (b) {
                ensure_grad(b);
                for (int j = 0; j < b->numel; j++) b->grad[j] += r->grad[split + j];
            }
            break;
        }

        case OP_NARROW: {
            /* r = parent[start..start+len), scatter grad back */
            int start = (int)e->scalar_arg;
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int j = 0; j < r->numel; j++) a->grad[start + j] += r->grad[j];
            }
            break;
        }

        case OP_MM: {
            /* r = a @ b where a=[m,n], b=[n,k], r=[m,k]
               d_a = grad @ b^T, d_b = a^T @ grad */
            int mm = a->shape[0], nn = a->shape[1], kk = r->shape[1];
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                /* d_a[i,j] = sum_p grad[i,p] * b[j,p] (grad @ b^T) */
                for (int i = 0; i < mm; i++)
                    for (int j = 0; j < nn; j++) {
                        double s = 0;
                        for (int p = 0; p < kk; p++) s += r->grad[i*kk+p] * b->data[j*kk+p];
                        a->grad[i*nn+j] += s;
                    }
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                /* d_b[j,p] = sum_i a[i,j] * grad[i,p] (a^T @ grad) */
                for (int j = 0; j < nn; j++)
                    for (int p = 0; p < kk; p++) {
                        double s = 0;
                        for (int i = 0; i < mm; i++) s += a->data[i*nn+j] * r->grad[i*kk+p];
                        b->grad[j*kk+p] += s;
                    }
            }
            break;
        }

        case OP_BMM: {
            /* r = a @ b where a=[B,m,n], b=[n,k], r=[B,m,k]
               d_a[bi] = grad[bi] @ b^T, d_b = sum_bi a[bi]^T @ grad[bi] */
            int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2], kk = b->shape[1];
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int bi = 0; bi < BB; bi++)
                    for (int i = 0; i < mm; i++)
                        for (int j = 0; j < nn; j++) {
                            double s = 0;
                            for (int p = 0; p < kk; p++)
                                s += r->grad[bi*mm*kk + i*kk+p] * b->data[j*kk+p];
                            a->grad[bi*mm*nn + i*nn+j] += s;
                        }
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                /* Accumulate across all batch elements */
                for (int bi = 0; bi < BB; bi++)
                    for (int j = 0; j < nn; j++)
                        for (int p = 0; p < kk; p++) {
                            double s = 0;
                            for (int i = 0; i < mm; i++)
                                s += a->data[bi*mm*nn + i*nn+j] * r->grad[bi*mm*kk + i*kk+p];
                            b->grad[j*kk+p] += s;
                        }
            }
            break;
        }

        case OP_BMM_3X3: {
            /* r = a @ b where a=[B,m,n], b=[B,n,k], r=[B,m,k]
               d_a[bi] = grad[bi] @ b[bi]^T, d_b[bi] = a[bi]^T @ grad[bi] */
            int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2], kk = b->shape[2];
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int bi = 0; bi < BB; bi++)
                    for (int i = 0; i < mm; i++)
                        for (int j = 0; j < nn; j++) {
                            double s = 0;
                            for (int p = 0; p < kk; p++)
                                s += r->grad[bi*mm*kk + i*kk+p] * b->data[bi*nn*kk + j*kk+p];
                            a->grad[bi*mm*nn + i*nn+j] += s;
                        }
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                for (int bi = 0; bi < BB; bi++)
                    for (int j = 0; j < nn; j++)
                        for (int p = 0; p < kk; p++) {
                            double s = 0;
                            for (int i = 0; i < mm; i++)
                                s += a->data[bi*mm*nn + i*nn+j] * r->grad[bi*mm*kk + i*kk+p];
                            b->grad[bi*nn*kk + j*kk+p] += s;
                        }
            }
            break;
        }

        case OP_SOFTMAX_3D: {
            /* r = softmax(a) on [B,m,n] along last dim. Same as 2D but B*m rows. */
            int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2];
            int total_rows = BB * mm;
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int i = 0; i < total_rows; i++) {
                    double dot = 0;
                    for (int j = 0; j < nn; j++)
                        dot += r->grad[i*nn+j] * r->data[i*nn+j];
                    for (int j = 0; j < nn; j++)
                        a->grad[i*nn+j] += r->data[i*nn+j] * (r->grad[i*nn+j] - dot);
                }
            }
            break;
        }

        case OP_TRANSPOSE_LAST2: {
            /* r = transpose_last2(a) where a=[B,m,n], r=[B,n,m]. Transpose grad back. */
            int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2];
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int bi = 0; bi < BB; bi++)
                    for (int i = 0; i < mm; i++)
                        for (int j = 0; j < nn; j++)
                            a->grad[bi*mm*nn + i*nn+j] += r->grad[bi*nn*mm + j*mm+i];
            }
            break;
        }

        case OP_TRANSPOSE_2D: {
            /* r = a^T where a=[m,n], r=[n,m]. Gradient: transpose back. */
            int mm = a->shape[0], nn = a->shape[1];
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int i = 0; i < mm; i++)
                    for (int j = 0; j < nn; j++)
                        a->grad[i*nn+j] += r->grad[j*mm+i];
            }
            break;
        }

        case OP_SOFTMAX_2D: {
            /* Row-wise softmax backward. For each row i:
               d_input[i,j] = sum_k (grad[i,k] * softmax[i,k] * (delta_jk - softmax[i,j])) */
            int mm = r->shape[0], nn = r->shape[1];
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int i = 0; i < mm; i++) {
                    double dot = 0;
                    for (int j = 0; j < nn; j++)
                        dot += r->grad[i*nn+j] * r->data[i*nn+j];
                    for (int j = 0; j < nn; j++)
                        a->grad[i*nn+j] += r->data[i*nn+j] * (r->grad[i*nn+j] - dot);
                }
            }
            break;
        }

        case OP_LOG_SOFTMAX_2D: {
            /* Row-wise log-softmax backward.
               d_input[i,j] = d_output[i,j] - softmax[i,j] * sum_k(d_output[i,k])
               where softmax[i,j] = exp(output[i,j]) since output = log_softmax */
            int mm = r->shape[0], nn = r->shape[1];
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int i = 0; i < mm; i++) {
                    double sum_grad = 0;
                    for (int j = 0; j < nn; j++)
                        sum_grad += r->grad[i*nn+j];
                    for (int j = 0; j < nn; j++)
                        a->grad[i*nn+j] += r->grad[i*nn+j] - exp(r->data[i*nn+j]) * sum_grad;
                }
            }
            break;
        }

        case OP_MASKED_FILL: {
            /* Gradient passes through where mask is 0, zero where mask is 1 */
            ensure_grad(r);
            if (a) {
                ensure_grad(a);
                for (int j = 0; j < a->numel; j++)
                    if (b->data[j] == 0.0) a->grad[j] += r->grad[j];
            }
            break;
        }

        case OP_LAYER_NORM_2D: {
            /* Row-wise layer norm backward.
               y = gamma * x_hat + beta, x_hat = (x - mean) / std
               d_gamma[j] = sum_i dy[i,j] * x_hat[i,j]
               d_beta[j]  = sum_i dy[i,j]
               dx = (1/std) * (dy*gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat)) */
            LayerNormMeta* meta = (LayerNormMeta*)e->op_meta;
            int mm = meta->m, nn = meta->n;
            ensure_grad(r);
            /* d_gamma and d_beta */
            if (meta->gamma && meta->gamma->requires_grad) {
                ensure_grad(meta->gamma);
                for (int j = 0; j < nn; j++) {
                    double dg = 0;
                    for (int i = 0; i < mm; i++) dg += r->grad[i*nn+j] * meta->x_hat[i*nn+j];
                    meta->gamma->grad[j] += dg;
                }
            }
            if (meta->bias && meta->bias->requires_grad) {
                ensure_grad(meta->bias);
                for (int j = 0; j < nn; j++) {
                    double db = 0;
                    for (int i = 0; i < mm; i++) db += r->grad[i*nn+j];
                    meta->bias->grad[j] += db;
                }
            }
            /* d_input */
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int i = 0; i < mm; i++) {
                    /* dx_hat = dy * gamma */
                    double mean_dxhat = 0;
                    double mean_dxhat_xhat = 0;
                    for (int j = 0; j < nn; j++) {
                        double dxh = r->grad[i*nn+j] * meta->gamma->data[j];
                        mean_dxhat += dxh;
                        mean_dxhat_xhat += dxh * meta->x_hat[i*nn+j];
                    }
                    mean_dxhat /= nn;
                    mean_dxhat_xhat /= nn;
                    for (int j = 0; j < nn; j++) {
                        double dxh = r->grad[i*nn+j] * meta->gamma->data[j];
                        a->grad[i*nn+j] += meta->rstd[i] *
                            (dxh - mean_dxhat - meta->x_hat[i*nn+j] * mean_dxhat_xhat);
                    }
                }
            }
            break;
        }

        case OP_MV: {
            /* d(Ax)/dA[i,j] = grad[i] * x[j], d(Ax)/dx[j] = sum_i A[i,j] * grad[i] */
            MvMeta* meta = (MvMeta*)e->op_meta;
            int m_mv = meta ? meta->m : a->shape[0];
            int n_mv = meta ? meta->n : a->shape[1];
            double* x_vals = meta ? meta->x_vals : b->data;
            ensure_grad(r);
            if (a->requires_grad) {
                ensure_grad(a);
                for (int ii = 0; ii < m_mv; ii++)
                    for (int jj = 0; jj < n_mv; jj++)
                        a->grad[ii*n_mv+jj] += r->grad[ii] * x_vals[jj];
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                for (int jj = 0; jj < n_mv; jj++) {
                    double s = 0;
                    for (int ii = 0; ii < m_mv; ii++) s += a->data[ii*n_mv+jj] * r->grad[ii];
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

        case OP_LOG_SOFTMAX: {
            /* log-softmax backward (1D): d_input[j] = grad[j] - exp(output[j]) * sum(grad) */
            if (a) {
                ensure_grad(a);
                ensure_grad(r);
                int n_ls = r->numel;
                double sum_grad = 0;
                for (int j = 0; j < n_ls; j++) sum_grad += r->grad[j];
                for (int j = 0; j < n_ls; j++)
                    a->grad[j] += r->grad[j] - exp(r->data[j]) * sum_grad;
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

        case OP_LSTM_GATES: {
            /* LSTM gates backward: propagate from hidden output to combined + prev_cell.
               hidden[j] = oG[j] * tanh(cell[j])
               cell[j] = fG[j] * prevCell[j] + iG[j] * gG[j]
               The cell output gradient needs to be collected separately.
               For now, this backward only handles the hidden output's gradient. */
            LstmGatesMeta* lm = (LstmGatesMeta*)e->op_meta;
            if (lm && a) {
                int o_lstm = lm->o;
                ensure_grad(a);  /* combined [4*o] */
                ensure_grad(r);  /* hidden [o] */
                if (b) ensure_grad(b);  /* prev_cell [o] */

                for (int j = 0; j < o_lstm; j++) {
                    double d_h = r->grad[j];
                    double tanhC = tanh(lm->new_cell[j]);

                    /* d_oGate = d_h * tanh(cell) */
                    double d_oG = d_h * tanhC;
                    /* d_cell from hidden path */
                    double d_cell = d_h * lm->oG[j] * (1.0 - tanhC * tanhC);

                    /* d_fGate = d_cell * prevCell */
                    double d_fG = d_cell * (b ? b->data[j] : 0);
                    /* d_iGate = d_cell * gG */
                    double d_iG = d_cell * lm->gG[j];
                    /* d_gGate = d_cell * iG */
                    double d_gG = d_cell * lm->iG[j];
                    /* d_prevCell = d_cell * fG */
                    if (b) b->grad[j] += d_cell * lm->fG[j];

                    /* Activation derivatives → combined gradient */
                    a->grad[j]          += d_iG * lm->iG[j] * (1.0 - lm->iG[j]);  /* sigmoid' */
                    a->grad[o_lstm + j]  += d_fG * lm->fG[j] * (1.0 - lm->fG[j]);
                    a->grad[2*o_lstm + j] += d_gG * (1.0 - lm->gG[j] * lm->gG[j]);  /* tanh' */
                    a->grad[3*o_lstm + j] += d_oG * lm->oG[j] * (1.0 - lm->oG[j]);
                }
            }
            break;
        }

        case OP_LSTM_GATES_CELL: {
            /* Cell output backward: cell[j] = fG[j]*prevCell[j] + iG[j]*gG[j]
               d_cell comes directly from downstream (FC layers reading cell state,
               and next timestep's LSTM using it as prev_cell). */
            LstmGatesMeta* lm = (LstmGatesMeta*)e->op_meta;
            if (lm && a) {
                int o_lstm = lm->o;
                ensure_grad(a);  /* combined [4*o] */
                ensure_grad(r);  /* cell [o] */
                if (b) ensure_grad(b);  /* prev_cell [o] */

                for (int j = 0; j < o_lstm; j++) {
                    double d_cell = r->grad[j];

                    /* d_fGate = d_cell * prevCell */
                    double d_fG = d_cell * (b ? b->data[j] : 0);
                    /* d_iGate = d_cell * gG */
                    double d_iG = d_cell * lm->gG[j];
                    /* d_gGate = d_cell * iG */
                    double d_gG = d_cell * lm->iG[j];
                    /* d_prevCell = d_cell * fG */
                    if (b) b->grad[j] += d_cell * lm->fG[j];

                    /* Activation derivatives → combined gradient (additive with OP_LSTM_GATES) */
                    a->grad[j]            += d_iG * lm->iG[j] * (1.0 - lm->iG[j]);
                    a->grad[o_lstm + j]    += d_fG * lm->fG[j] * (1.0 - lm->fG[j]);
                    a->grad[2*o_lstm + j]  += d_gG * (1.0 - lm->gG[j] * lm->gG[j]);
                    /* No output gate gradient from cell path (oG only affects hidden) */
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

        case OP_NTM_READ_HEAD: {
            NtmReadHeadMeta* m = (NtmReadHeadMeta*)e->op_meta;
            if (m) {
                ensure_grad(r);
                ntm_read_head_backward_chain(m, r->grad);
            }
            break;
        }

        case OP_NTM_READ_HEAD_READ: {
            NtmReadHeadMeta* m = (NtmReadHeadMeta*)e->op_meta;
            if (m) {
                ensure_grad(r);
                int nn = m->n, ww = m->w;
                /* d_focused[i] = sum_j d_read_out[j] * memory[i,j] */
                double* d_focused = calloc(nn, sizeof(double));
                for (int ii = 0; ii < nn; ii++)
                    for (int jj = 0; jj < ww; jj++)
                        d_focused[ii] += r->grad[jj] * m->memory->data[ii * ww + jj];
                /* d_memory[i,j] += d_read_out[j] * focused[i] */
                if (m->memory->requires_grad) {
                    ensure_grad(m->memory);
                    for (int ii = 0; ii < nn; ii++)
                        for (int jj = 0; jj < ww; jj++)
                            m->memory->grad[ii * ww + jj] += r->grad[jj] * m->focused[ii];
                }
                ntm_read_head_backward_chain(m, d_focused);
                free(d_focused);
            }
            break;
        }

        case OP_NTM_INTERP_WRITE: {
            NtmInterpWriteMeta* m = (NtmInterpWriteMeta*)e->op_meta;
            if (m) {
                ensure_grad(r);
                int nn = m->n, ww = m->w;
                /* d_memory = d_result */
                if (a && a->requires_grad) {
                    ensure_grad(a);
                    for (int ij = 0; ij < nn * ww; ij++)
                        a->grad[ij] += r->grad[ij];
                }
                /* d_weights[i] = sum_j d_result[i,j] * add_vector[j] */
                if (b && b->requires_grad) {
                    ensure_grad(b);
                    for (int ii = 0; ii < nn; ii++) {
                        double s = 0;
                        for (int jj = 0; jj < ww; jj++)
                            s += r->grad[ii * ww + jj] * m->add_vector->data[jj];
                        b->grad[ii] += s;
                    }
                }
                /* d_add_vector[j] = sum_i d_result[i,j] * weights[i] */
                if (m->add_vector->requires_grad) {
                    ensure_grad(m->add_vector);
                    for (int jj = 0; jj < ww; jj++) {
                        double s = 0;
                        for (int ii = 0; ii < nn; ii++)
                            s += r->grad[ii * ww + jj] * b->data[ii];
                        m->add_vector->grad[jj] += s;
                    }
                }
            }
            break;
        }

        case OP_EMBEDDING: {
            /* Scatter grad rows back to weight matrix */
            EmbeddingMeta* meta = (EmbeddingMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int i = 0; i < meta->n; i++) {
                    int idx = meta->indices[i];
                    for (int j = 0; j < meta->embedDim; j++)
                        a->grad[idx * meta->embedDim + j] += r->grad[i * meta->embedDim + j];
                }
            }
            break;
        }

        case OP_BATCH_NORM: {
            /* y = gamma * x_hat + beta where x_hat = (x - mean) / std
               d_gamma[c] = sum_j dy[c,j] * x_hat[c,j]
               d_beta[c]  = sum_j dy[c,j]
               dx = rstd * (dy*gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat)) */
            BatchNormMeta* meta = (BatchNormMeta*)e->op_meta;
            int CC = meta->C, sp = meta->spatial;
            ensure_grad(r);
            /* d_gamma and d_beta */
            if (meta->gamma->requires_grad) {
                ensure_grad(meta->gamma);
                for (int c = 0; c < CC; c++) {
                    double dg = 0;
                    for (int j = 0; j < sp; j++) dg += r->grad[c*sp+j] * meta->x_hat[c*sp+j];
                    meta->gamma->grad[c] += dg;
                }
            }
            if (meta->beta->requires_grad) {
                ensure_grad(meta->beta);
                for (int c = 0; c < CC; c++) {
                    double db = 0;
                    for (int j = 0; j < sp; j++) db += r->grad[c*sp+j];
                    meta->beta->grad[c] += db;
                }
            }
            /* d_input */
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int c = 0; c < CC; c++) {
                    double mean_dxhat = 0, mean_dxhat_xhat = 0;
                    for (int j = 0; j < sp; j++) {
                        double dxh = r->grad[c*sp+j] * meta->gamma->data[c];
                        mean_dxhat += dxh;
                        mean_dxhat_xhat += dxh * meta->x_hat[c*sp+j];
                    }
                    mean_dxhat /= sp;
                    mean_dxhat_xhat /= sp;
                    for (int j = 0; j < sp; j++) {
                        double dxh = r->grad[c*sp+j] * meta->gamma->data[c];
                        a->grad[c*sp+j] += meta->rstd[c] *
                            (dxh - mean_dxhat - meta->x_hat[c*sp+j] * mean_dxhat_xhat);
                    }
                }
            }
            break;
        }

        case OP_DROPOUT: {
            /* Gradient passes through the same mask used in forward */
            DropoutMeta* meta = (DropoutMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int j = 0; j < meta->numel; j++)
                    a->grad[j] += r->grad[j] * meta->mask[j];
            }
            break;
        }

        case OP_AVG_POOL1D: {
            AvgPool1DMeta* meta = (AvgPool1DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                double scale = 1.0 / meta->kL;
                for (int c = 0; c < meta->C; c++)
                    for (int ol = 0; ol < meta->oL; ol++)
                        for (int kl = 0; kl < meta->kL; kl++)
                            a->grad[c*meta->L + ol*meta->stride + kl] += r->grad[c*meta->oL + ol] * scale;
            }
            break;
        }

        case OP_AVG_POOL2D: {
            AvgPool2DMeta* meta = (AvgPool2DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                double scale = 1.0 / (meta->kH * meta->kW);
                for (int c = 0; c < meta->C; c++)
                    for (int oh = 0; oh < meta->oH; oh++)
                        for (int ow = 0; ow < meta->oW; ow++)
                            for (int kh = 0; kh < meta->kH; kh++)
                                for (int kw = 0; kw < meta->kW; kw++)
                                    a->grad[c*meta->H*meta->W + (oh*meta->strH+kh)*meta->W + ow*meta->strW+kw]
                                        += r->grad[c*meta->oH*meta->oW + oh*meta->oW + ow] * scale;
            }
            break;
        }

        case OP_CONV1D: {
            Conv1DMeta* meta = (Conv1DMeta*)e->op_meta;
            int inC = meta->inC, outC = meta->outC, LL = meta->L;
            int kL = meta->kL, pad = meta->pad, str = meta->stride, oL = meta->oL;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int oc = 0; oc < outC; oc++)
                    for (int ol = 0; ol < oL; ol++) {
                        double dout = r->grad[oc*oL + ol];
                        for (int ic = 0; ic < inC; ic++)
                            for (int kl = 0; kl < kL; kl++) {
                                int il = ol * str - pad + kl;
                                if (il >= 0 && il < LL)
                                    a->grad[ic*LL + il] += dout * b->data[oc*inC*kL + ic*kL + kl];
                            }
                    }
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                for (int oc = 0; oc < outC; oc++)
                    for (int ic = 0; ic < inC; ic++)
                        for (int kl = 0; kl < kL; kl++) {
                            double s = 0;
                            for (int ol = 0; ol < oL; ol++) {
                                int il = ol * str - pad + kl;
                                if (il >= 0 && il < LL)
                                    s += r->grad[oc*oL + ol] * a->data[ic*LL + il];
                            }
                            b->grad[oc*inC*kL + ic*kL + kl] += s;
                        }
            }
            Tensor* bias_t = (Tensor*)e->inputs;
            if (bias_t && bias_t->requires_grad) {
                ensure_grad(bias_t);
                for (int oc = 0; oc < outC; oc++) {
                    double s = 0;
                    for (int ol = 0; ol < oL; ol++) s += r->grad[oc*oL + ol];
                    bias_t->grad[oc] += s;
                }
            }
            break;
        }

        case OP_MAX_POOL1D: {
            MaxPool1DMeta* meta = (MaxPool1DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int out_numel = meta->C * meta->oL;
                for (int i = 0; i < out_numel; i++)
                    a->grad[meta->max_indices[i]] += r->grad[i];
            }
            break;
        }

        case OP_CONV2D: {
            /* r = conv2d(a=input, b=kernel) + bias
               a=[inC,H,W], b=[outC,inC,kH,kW], r=[outC,oH,oW] */
            Conv2DMeta* meta = (Conv2DMeta*)e->op_meta;
            int inC = meta->inC, outC = meta->outC;
            int HH = meta->H, WW = meta->W, kH = meta->kH, kW = meta->kW;
            int padH = meta->padH, padW = meta->padW;
            int strideH = meta->strH, strideW = meta->strW;
            int oH = meta->oH, oW = meta->oW;
            ensure_grad(r);

            /* d_input */
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int oc = 0; oc < outC; oc++)
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++) {
                            double dout = r->grad[oc*oH*oW + oh*oW + ow];
                            for (int ic = 0; ic < inC; ic++)
                                for (int kh = 0; kh < kH; kh++)
                                    for (int kw = 0; kw < kW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < HH && iw >= 0 && iw < WW)
                                            a->grad[ic*HH*WW + ih*WW + iw] +=
                                                dout * b->data[oc*inC*kH*kW + ic*kH*kW + kh*kW + kw];
                                    }
                        }
            }

            /* d_kernel */
            if (b && b->requires_grad) {
                ensure_grad(b);
                for (int oc = 0; oc < outC; oc++)
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                double s = 0;
                                for (int oh = 0; oh < oH; oh++)
                                    for (int ow = 0; ow < oW; ow++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < HH && iw >= 0 && iw < WW)
                                            s += r->grad[oc*oH*oW + oh*oW + ow]
                                               * a->data[ic*HH*WW + ih*WW + iw];
                                    }
                                b->grad[oc*inC*kH*kW + ic*kH*kW + kh*kW + kw] += s;
                            }
            }

            /* d_bias */
            Tensor* bias_t = (Tensor*)e->inputs;  /* stored in inputs field */
            if (bias_t && bias_t->requires_grad) {
                ensure_grad(bias_t);
                for (int oc = 0; oc < outC; oc++) {
                    double s = 0;
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++)
                            s += r->grad[oc*oH*oW + oh*oW + ow];
                    bias_t->grad[oc] += s;
                }
            }
            break;
        }

        case OP_MAX_POOL2D: {
            /* r = max_pool2d(a=input). Gradient flows only to max positions. */
            MaxPool2DMeta* meta = (MaxPool2DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int out_numel = meta->C * meta->oH * meta->oW;
                for (int i = 0; i < out_numel; i++)
                    a->grad[meta->max_indices[i]] += r->grad[i];
            }
            break;
        }

        default: break; /* unimplemented backward */
        }
    }
    (void)processed; (void)skipped;
    prof_backward_ms += _wall_ms() - t0;
    prof_backward_ops += processed;
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
    int rg = combined->requires_grad || prev_cell->requires_grad;
    double* out_hidden = calloc(o, sizeof(double));
    double* out_cell = calloc(o, sizeof(double));

    /* Save gate activations for backward */
    LstmGatesMeta* meta = NULL;
    if (rg) {
        meta = arena_alloc(sizeof(LstmGatesMeta));
        meta->o = o;
        meta->iG = arena_alloc(o * sizeof(double));
        meta->fG = arena_alloc(o * sizeof(double));
        meta->gG = arena_alloc(o * sizeof(double));
        meta->oG = arena_alloc(o * sizeof(double));
        meta->new_cell = arena_alloc(o * sizeof(double));
    }

    for (int j = 0; j < o; j++) {
        double ig = 1.0 / (1.0 + exp(-combined->data[j]));
        double fg = 1.0 / (1.0 + exp(-combined->data[o+j]));
        double gg = tanh(combined->data[2*o+j]);
        double og = 1.0 / (1.0 + exp(-combined->data[3*o+j]));
        out_cell[j] = fg * prev_cell->data[j] + ig * gg;
        out_hidden[j] = og * tanh(out_cell[j]);
        if (meta) {
            meta->iG[j] = ig; meta->fG[j] = fg;
            meta->gG[j] = gg; meta->oG[j] = og;
            meta->new_cell[j] = out_cell[j];
        }
    }

    int shape[] = {o};
    *out_h = make_tensor(out_hidden, shape, 1, rg);
    *out_c = make_tensor(out_cell, shape, 1, rg);
    free(out_hidden);
    free(out_cell);

    if (rg) {
        /* Record hidden output with OP_LSTM_GATES — backward propagates d_hidden */
        int idx_h = tape_append(OP_LSTM_GATES, (Tensor*)*out_h, combined, prev_cell, (double)o);
        tape[idx_h].op_meta = meta;
        /* Record cell output with OP_LSTM_GATES_CELL — backward propagates d_cell.
           Both entries share the same metadata and accumulate gradients additively
           into combined->grad and prev_cell->grad. */
        int idx_c = tape_append(OP_LSTM_GATES_CELL, (Tensor*)*out_c, combined, prev_cell, (double)o);
        tape[idx_c].op_meta = meta;
    }
}

TensorPair* tensor_lstm_gates_pair(TensorHandle combined, TensorHandle prev_cell, int o) {
    TensorPair* p = arena_alloc(sizeof(TensorPair));
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

double param_grad_item_at(int param_idx, int elem_idx) {
    Tensor* t = param_registry[param_idx].tensor;
    if (!t->grad || elem_idx >= t->numel) return 0.0;
    return t->grad[elem_idx];
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

void param_load_data(int idx, const double* data, int numel) {
    Tensor* t = param_registry[idx].tensor;
    if (t->numel != numel) {
        fprintf(stderr, "param_load_data: size mismatch for '%s': expected %d, got %d\n",
                param_registry[idx].name, t->numel, numel);
        return;
    }
    memcpy(t->data, data, numel * sizeof(double));
}

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    Tensor* t = (Tensor*)h;
    for (int i = 0; i < t->numel; i++) t->data[i] -= val;
    return h;
}

/* ================================================================
   Convenience functions
   ================================================================ */

/* Create a one-hot encoded 1D tensor from token indices.
   tokens: array of token indices (int), n_tokens long
   vocab_size: number of classes per token
   Output: 1D tensor of length n_tokens * vocab_size */
TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size) {
    int total = n_tokens * vocab_size;
    double* data = calloc(total, sizeof(double));  /* zeros */
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            data[i * vocab_size + tok] = 1.0;
    }
    int shape[] = {total};
    Tensor* r = make_tensor(data, shape, 1, 0);
    free(data);
    free(tokens);
    return r;
}

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    int shape[] = {n};
    TensorHandle t = tensor_create(data, shape, 1, requires_grad);
    free(data);  /* tensor_create copies data into arena; free the original */
    return t;
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    int shape[] = {rows, cols};
    TensorHandle t = tensor_create(data, shape, 2, requires_grad);
    free(data);
    return t;
}

TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
    int shape[] = {rows, cols};
    return tensor_reshape(h, shape, 2);
}

double* tensor_alloc_doubles(int n) { return calloc(n, sizeof(double)); }
void tensor_free_doubles(double* buf) { free(buf); }
double tensor_read_double(double* buf, int idx) { return buf[idx]; }
void tensor_write_double(double* buf, int idx, double val) { buf[idx] = val; }

TensorHandle* tensor_ptr_array_alloc(int n) {
    return calloc(n, sizeof(TensorHandle));
}

void tensor_ptr_array_set(TensorHandle* arr, int idx, TensorHandle t) {
    arr[idx] = t;
}

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    /* Fast path: if all inputs are consecutive selects from the same parent
       tensor (data pointers are contiguous), skip the copy and return a
       tensor that shares the parent's data. This eliminates the repack
       cost when tensorToScalars → vecStackTensor round-trips. */
    if (count > 0) {
        Tensor* first = (Tensor*)arr[0];
        double* base = first->data;
        int consecutive = 1;
        int rg_check = first->requires_grad;
        for (int i = 1; i < count; i++) {
            Tensor* t = (Tensor*)arr[i];
            if (t->data != base + i) { consecutive = 0; break; }
            if (t->requires_grad) rg_check = 1;
        }
        if (consecutive) { 
            /* Create a tensor that shares the parent's data buffer (no copy) */
            Tensor* r = arena_alloc(sizeof(Tensor));
            memset(r, 0, sizeof(Tensor));
            r->data = base;  /* shared with parent */
            r->shape = arena_alloc(sizeof(int));
            r->shape[0] = count;
            r->rank = 1;
            r->numel = count;
            r->requires_grad = rg_check;
            r->persistent = 0;
            /* Still record OP_STACK with input pointers for backward.
               STACK backward distributes r->grad[i] to inputs[i]->grad[0].
               The inputs are SELECT views, so their grad flows to the parent. */
            if (rg_check) {
                Tensor** inputs = malloc(count * sizeof(Tensor*));
                for (int i = 0; i < count; i++) inputs[i] = (Tensor*)arr[i];
                int idx = tape_append(OP_STACK, r, NULL, NULL, 0);
                tape[idx].inputs = inputs;
                tape[idx].input_count = count;
            }
            free(arr);
            return r;
        }
    }

    /* Slow path: copy values and create new tensor */
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
    free(data);  /* free the input buffer (caller used tensor_alloc_doubles) */
    t->shape = malloc(2 * sizeof(int));
    t->shape[0] = rows; t->shape[1] = cols;
    t->rank = 2; t->numel = numel;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    int numel = d0 * d1 * d2 * d3;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);
    t->shape = malloc(4 * sizeof(int));
    t->shape[0] = d0; t->shape[1] = d1; t->shape[2] = d2; t->shape[3] = d3;
    t->rank = 4; t->numel = numel;
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
    free(data);
    t->shape = malloc(sizeof(int));
    t->shape[0] = n;
    t->rank = 1; t->numel = n;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

/* Persistent tensors WITHOUT requires_grad — for non-learnable NTM state */
TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    int numel = rows * cols;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);
    t->shape = malloc(2 * sizeof(int));
    t->shape[0] = rows; t->shape[1] = cols;
    t->rank = 2; t->numel = numel;
    t->requires_grad = 0;
    t->tape_idx = -1;
    t->persistent = 1;
    return t;
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(n * sizeof(double));
    memcpy(t->data, data, n * sizeof(double));
    free(data);
    t->shape = malloc(sizeof(int));
    t->shape[0] = n;
    t->rank = 1; t->numel = n;
    t->requires_grad = 0;
    t->tape_idx = -1;
    t->persistent = 1;
    return t;
}

TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    Tensor* t = (Tensor*)h;
    int cols = t->shape[1];
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

/* Compute total number of elements across all params (for per-element optimizer buffers) */
static int param_total_elements(void) {
    int total = 0;
    for (int i = 0; i < param_count_val; i++)
        total += param_registry[i].tensor->numel;
    return total;
}

/* Offset into the flat per-element buffer for param i, element j */
static int param_element_offset(int param_idx) {
    int off = 0;
    for (int i = 0; i < param_idx; i++)
        off += param_registry[i].tensor->numel;
    return off;
}

static void optimizer_ensure_buffers(Optimizer* opt) {
    if (opt->allocated) return;
    int n = param_total_elements();
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

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 3;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->weight_decay = weight_decay;
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
    double t0_opt = _wall_ms();
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    opt->t++;

    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        int base = param_element_offset(i);

        for (int j = 0; j < t->numel; j++) {
            double g = t->grad[j];
            int idx = base + j;  /* per-element index into optimizer buffers */

            switch (opt->type) {
            case 0: /* SGD */
                t->data[j] -= opt->lr * g;
                break;

            case 1: { /* RMSprop */
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
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                t->data[j] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
                break;
            }

            case 3: { /* AdamW (decoupled weight decay) */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                t->data[j] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
                t->data[j] -= opt->lr * opt->weight_decay * t->data[j];
                break;
            }
            }
        }
    }

    /* Snapshot tape size before reset */
    prof_forward_ops = tape_size;

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
    prof_optimizer_ms += _wall_ms() - t0_opt;
    prof_epochs++;
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
   Optimizer buffer accessors (for serialization)
   ================================================================ */

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return param_count_val;
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, param_registry[idx].tensor->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(out, opt->m + offset, numel * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, param_registry[idx].tensor->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(out, opt->v + offset, numel * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(opt->m + offset, data, numel * sizeof(double));
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(opt->v + offset, data, numel * sizeof(double));
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
    Optimizer* opt = (Optimizer*)h;
    out9[0] = (double)opt->type;
    out9[1] = opt->lr;
    out9[2] = opt->beta1;
    out9[3] = opt->beta2;
    out9[4] = opt->eps;
    out9[5] = opt->alpha;
    out9[6] = opt->weight_decay;
    out9[7] = opt->momentum;
    out9[8] = (double)opt->t;
}

void optimizer_set_meta(OptimizerHandle h, const double* in9) {
    Optimizer* opt = (Optimizer*)h;
    opt->type = (int)in9[0];
    opt->lr = in9[1];
    opt->beta1 = in9[2];
    opt->beta2 = in9[3];
    opt->eps = in9[4];
    opt->alpha = in9[5];
    opt->weight_decay = in9[6];
    opt->momentum = in9[7];
    opt->t = (int)in9[8];
}

/* ================================================================
   System
   ================================================================ */

int backend_supports_tensor_params(void) { return 1; }

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

void backend_reset_for_eval(void) {
    tape_reset();
    /* Re-register params so they have valid tape indices */
    for (int j = 0; j < param_count_val; j++) {
        Tensor* t = param_registry[j].tensor;
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

void backend_memory_report(void) {
    /* Arena stats */
    int chunk_count = 0;
    size_t total_cap = 0, total_used = 0;
    for (ArenaChunk* c = arena_head; c; c = c->next) {
        chunk_count++;
        total_cap += c->cap;
        total_used += c->used;
    }

    /* Param stats */
    int total_param_elems = 0;
    size_t param_grad_bytes = 0;
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        total_param_elems += t->numel;
        if (t->grad) param_grad_bytes += t->numel * sizeof(double);
    }

    size_t leaked_bytes = (size_t)persistent_scalar_count * 56;  /* ~56B per scalar */

    fprintf(stderr, "=== Memory Report ===\n");
    fprintf(stderr, "  Arena: %d chunks, %zuKB capacity, %zuKB used\n",
            chunk_count, total_cap / 1024, total_used / 1024);
    fprintf(stderr, "  Tape: %d entries (cap %d), %zuKB\n",
            tape_size, tape_cap, (size_t)tape_cap * sizeof(TapeEntry) / 1024);
    fprintf(stderr, "  Params: %d tensors, %d elements, %zuKB grads\n",
            param_count_val, total_param_elems, param_grad_bytes / 1024);
    fprintf(stderr, "  Persistent scalars: %d (~%zuKB leaked)\n",
            persistent_scalar_count, leaked_bytes / 1024);
    fprintf(stderr, "  RSS: peak=%dMB cur=%dMB\n",
            get_rss_mb(), get_current_rss_mb());
    fprintf(stderr, "  Expected: arena %zuKB + tape %zuKB + params %zuKB + leaked %zuKB = %zuKB\n",
            total_cap / 1024,
            (size_t)tape_cap * sizeof(TapeEntry) / 1024,
            (size_t)total_param_elems * sizeof(double) / 1024,
            leaked_bytes / 1024,
            (total_cap + (size_t)tape_cap * sizeof(TapeEntry) +
             (size_t)total_param_elems * sizeof(double) + leaked_bytes) / 1024);
}

/* ================================================================
   Profiling
   ================================================================ */

void backend_profile_reset(void) {
    prof_forward_ms = prof_backward_ms = prof_optimizer_ms = 0;
    prof_forward_ops = prof_backward_ops = prof_epochs = 0;
}

void backend_profile_report(void) {
    fprintf(stderr, "=== Profile Report ===\n");
    fprintf(stderr, "  Epochs: %d\n", prof_epochs);
    fprintf(stderr, "  Tape entries (last fwd): %d\n", tape_size);
    fprintf(stderr, "  Params: %d tensors, %d elements\n",
            param_count_val, ({int n=0; for(int i=0;i<param_count_val;i++) n+=param_registry[i].tensor->numel; n;}));
    fprintf(stderr, "  Forward:   %.1fms total (%.1fms/epoch)\n",
            prof_forward_ms, prof_epochs > 0 ? prof_forward_ms / prof_epochs : 0);
    fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n",
            prof_backward_ms, prof_epochs > 0 ? prof_backward_ms / prof_epochs : 0);
    fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n",
            prof_optimizer_ms, prof_epochs > 0 ? prof_optimizer_ms / prof_epochs : 0);
    double total = prof_forward_ms + prof_backward_ms + prof_optimizer_ms;
    fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n",
            total, prof_epochs > 0 ? total / prof_epochs : 0);
}

/* ================================================================
   Debug
   ================================================================ */

const char* backend_name(void) { return "tape"; }

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

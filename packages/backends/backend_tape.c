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

#define ARENA_INIT_SIZE (1 << 22)  /* 4 MB */

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

/* Like make_tensor but data is ALREADY arena-allocated — no copy needed.
   Caller must have arena_alloc'd the data buffer. */
static Tensor* make_tensor_arena(double* arena_data, int numel, int* shape, int rank, int requires_grad) {
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    t->data = arena_data;  /* already in arena — no copy */
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
    OP_MV, OP_LINEAR, OP_DOT, OP_OUTER,
    OP_SOFTMAX, OP_LOG_SOFTMAX,
    OP_SUM, OP_MEAN,
    OP_BCE_WITH_LOGITS,
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
    OP_GRU_CELL,      /* GRU cell: z,r,n gates -> new hidden */
    OP_EMBEDDING,     /* row gather from weight matrix */
    OP_BATCH_NORM,    /* per-channel normalization */
    OP_DROPOUT,       /* inverted dropout with stored mask */
    OP_AVG_POOL1D,    /* [C,L] -> [C,oL] mean pooling */
    OP_AVG_POOL2D,    /* [C,H,W] -> [C,oH,oW] mean pooling */
    OP_CONV1D,        /* [inC,L] * [outC,inC,kL] + [outC] -> [outC,oL] */
    OP_MAX_POOL1D,    /* [C,L] -> [C,oL] with max indices */
    OP_CONV2D,        /* [inC,H,W] * [outC,inC,kH,kW] + [outC] -> [outC,oH,oW] */
    OP_CONV2D_BATCHED,    /* [B,inC,H,W] * [outC,inC,kH,kW] + [outC] -> [B,outC,oH,oW] */
    OP_MAX_POOL2D,    /* [C,H,W] -> [C,oH,oW] with max indices */
    OP_MAX_POOL2D_BATCHED,    /* [B,C,H,W] -> [B,C,oH,oW] with max indices */
    OP_CUMPROD,       /* cumulative product along dim 0 */
    OP_GATHER,        /* gather by index: out[i] = input[index[i]] */
    OP_SCATTER_ADD,   /* scatter add: out[index[i]] += src[i] */
    OP_LEAKY_RELU,    /* max(alpha*x, x) — alpha in scalar_arg */
    OP_SILU,          /* x * sigmoid(x) (Swish activation) */
    OP_LINEAR_2D,     /* [B,o] = [B,i] @ [o,i]^T + [o] (batched fused linear) */
    OP_CONCAT_2D_AXIS1, /* [m,n] ++ [m,k] -> [m,n+k] along axis 1 */
    OP_SOFTPLUS,      /* log(1 + exp(x)), backward = sigmoid(x) */
    OP_TILE_2D,       /* [m,n] -> [m*rep0, n*rep1]; reps in scalar_arg via 2 int fields */
    OP_COUNT          /* sentinel — must be last */
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
typedef struct { int m, n; double* x_vals; Tensor* bias; } LinearMeta;
typedef struct { int B, i, o; double* x_vals; Tensor* bias; } Linear2dMeta;
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
    Tensor* gamma;     /* scale parameter [n] */
    Tensor* bias;      /* shift parameter [n] */
    double* x_hat;     /* normalized values [m*n] */
    double* rstd;      /* reciprocal std devs [m] */
    int m, n;
} LayerNormMeta;

typedef struct {
    int o;
    double* zG; double* rG; double* nG;  /* activated gate values [o] each */
    Tensor* prev;                         /* prev hidden state [o] (for backward) */
} GruCellMeta;

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
    int B, inC, outC, H, W, kH, kW, padH, padW, strH, strW, oH, oW;
} Conv2DBatchedMeta;

typedef struct {
    int C, H, W, kH, kW, strH, strW, oH, oW;
    int* max_indices;  /* [C * oH * oW] index into flat input per-channel */
} MaxPool2DMeta;

typedef struct {
    int B, C, H, W, kH, kW, strH, strW, oH, oW;
    int* max_indices;  /* [B * C * oH * oW] flat-input index per (b, c, oh, ow) */
} MaxPool2DBatchedMeta;

/* ================================================================
   Generic typed chunked arena.

   Append-only collection of fixed-size T elements stored as a linked
   list of equal-capacity chunks. Append returns a pointer to the new
   element; index lookup is O(size / chunk_capacity).

   Why: realloc-based growth requires free(old_buffer), and free here
   has caused use-after-free SIGSEGVs (the long-running tape-realloc
   bug — leaking the old tape ran cleanly, the actual stale reader was
   never pinpointed). Chunks here are never freed or moved mid-life, so
   any pointer returned by typed_arena_append stays valid until
   typed_arena_reset is called explicitly.

   Sister structure: ArenaChunk above is the variable-size byte-level
   arena (used for tensor data, op_meta, etc.). This is its
   fixed-element-size cousin for index-based collections like the tape.
   ================================================================ */

typedef struct TypedArenaChunk {
    void* data;                       /* element_size * chunk_capacity bytes */
    struct TypedArenaChunk* next;
} TypedArenaChunk;

typedef struct TypedArena {
    TypedArenaChunk* head;            /* first chunk, allocated lazily */
    TypedArenaChunk* tail;            /* chunk receiving the next append */
    int size;                         /* total elements across all chunks */
    int tail_count;                   /* elements written in tail (0..chunk_capacity) */
    int chunk_capacity;               /* configured: elements per chunk */
    size_t element_size;              /* configured: sizeof(T) */
} TypedArena;

static void* typed_arena_append(TypedArena* a) {
    if (!a->head) {
        a->head = malloc(sizeof(TypedArenaChunk));
        a->head->data = calloc(a->chunk_capacity, a->element_size);
        a->head->next = NULL;
        a->tail = a->head;
        a->tail_count = 0;
    } else if (a->tail_count == a->chunk_capacity) {
        if (a->tail->next) {
            a->tail = a->tail->next;
        } else {
            TypedArenaChunk* c = malloc(sizeof(TypedArenaChunk));
            c->data = calloc(a->chunk_capacity, a->element_size);
            c->next = NULL;
            a->tail->next = c;
            a->tail = c;
        }
        a->tail_count = 0;
    }
    void* p = (char*)a->tail->data + a->tail_count * a->element_size;
    a->size++;
    a->tail_count++;
    return p;
}

static void* typed_arena_at(TypedArena* a, int idx) {
    int chunk_idx = idx / a->chunk_capacity;
    int intra = idx % a->chunk_capacity;
    TypedArenaChunk* c = a->head;
    while (chunk_idx-- > 0 && c) c = c->next;
    return c ? (char*)c->data + intra * a->element_size : NULL;
}

/* Resets size to 0 but keeps chunks allocated for reuse on the next
   recording. Caller is responsible for any per-element teardown
   (heap pointers stored in entries) BEFORE calling reset. */
static void typed_arena_reset(TypedArena* a) {
    a->size = 0;
    a->tail = a->head;
    a->tail_count = 0;
}

/* ================================================================
   Tape — autograd Wengert list.

   Implemented as a TypedArena<TapeEntry> with 64K-entry chunks
   (~5 MB at sizeof(TapeEntry) ~ 80 bytes). Forward ops append; backward
   walks tape entries in reverse; optimizer_step calls tape_reset which
   tears down per-entry heap allocations and resets size to 0 without
   freeing chunks (so the next forward reuses them — no malloc churn).
   ================================================================ */

#define TAPE_CHUNK_SIZE (1 << 16)

static TypedArena tape_arena = {
    .head = NULL, .tail = NULL, .size = 0, .tail_count = 0,
    .chunk_capacity = TAPE_CHUNK_SIZE,
    .element_size = sizeof(TapeEntry),
};

#define tape_size (tape_arena.size)

static inline TapeEntry* tape_at(int idx) {
    return (TapeEntry*)typed_arena_at(&tape_arena, idx);
}

/* fwd-decls: definitions are in the profiling section. */
static double _wall_ms(void);
extern double prof_forward_per_op[];
extern int prof_forward_count_per_op[];
extern double prof_op_t_prev;

/* When > 0, tape_append is a no-op and any tensor created inside is
   marked requires_grad=0. Used by withNoGrad: rollouts, evals, any
   forward that doesn't need gradients. Counter (not bool) so nested
   withNoGrad scopes nest correctly. Mirrors PyTorch's torch.no_grad().
   The previous tensor_no_grad_begin/end were stubs; now wired up. */
static int no_grad_depth = 0;
/* Dummy tape entry returned by tape_append in no_grad mode. Many
   callers do `e = tape_append(...); e->op_meta = ...;` — they need
   a valid (non-null) pointer to write to. We give them this static
   buffer; the writes are scratch and never read (the result tensor
   has tape_idx=-1 so backward never reaches it). */
static TapeEntry _no_grad_dummy_entry;

static TapeEntry* tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
    /* Inside withNoGrad: skip the tape entry entirely and mark the
       result as not grad-tracked, so downstream ops don't propagate
       grad through it. Saves both forward overhead (no entry) and
       backward overhead (fewer entries to walk). */
    if (no_grad_depth > 0) {
        if (result) {
            result->requires_grad = 0;
            result->tape_idx = -1;
        }
        /* Return a writable dummy so callers that do
              e = tape_append(...); e->op_meta = ...;
           don't crash. The result has tape_idx=-1 so backward never
           dereferences this entry — the writes are scratch. */
        memset(&_no_grad_dummy_entry, 0, sizeof(_no_grad_dummy_entry));
        return &_no_grad_dummy_entry;
    }
    /* Attribute the time since the previous tape_append (or epoch_begin)
       to this op — covers its compute + setup. Idris-side glue between
       ops adds small noise, but at typical 30+ µs per op the signal
       still surfaces a 2× regression cleanly. */
    if (prof_op_t_prev > 0 && op >= 0 && op < OP_COUNT) {
        double now = _wall_ms();
        prof_forward_per_op[op] += now - prof_op_t_prev;
        prof_forward_count_per_op[op]++;
        prof_op_t_prev = now;
    }
    TapeEntry* e = (TapeEntry*)typed_arena_append(&tape_arena);
    memset(e, 0, sizeof(TapeEntry));
    e->op = op;
    e->result = result;
    e->arg1 = arg1;
    e->arg2 = arg2;
    e->scalar_arg = scalar_arg;
    if (result) result->tape_idx = tape_arena.size - 1;
    return e;
}

static void tape_reset(void) {
    /* Walk chunks tearing down per-entry heap allocations before reset.
       Each entry's op_meta and inputs were heap-allocated by the forward
       op and must be freed before the entry is reused on the next epoch. */
    int remaining = tape_arena.size;
    for (TypedArenaChunk* c = tape_arena.head; c && remaining > 0; c = c->next) {
        int n = remaining > tape_arena.chunk_capacity ? tape_arena.chunk_capacity : remaining;
        TapeEntry* entries = (TapeEntry*)c->data;
        for (int i = 0; i < n; i++) {
            TapeEntry* e = &entries[i];
        /* Free OP_STACK inputs arrays */
        if (e->op == OP_STACK && e->inputs) {
            free(e->inputs);
            e->inputs = NULL;
        }
        /* Free OP_LAYER_NORM_2D heap arrays */
        if (e->op == OP_LAYER_NORM_2D && e->op_meta) {
            LayerNormMeta* meta = (LayerNormMeta*)e->op_meta;
            free(meta->x_hat);
            free(meta->rstd);
            meta->x_hat = NULL;
            meta->rstd = NULL;
        }
        /* Free OP_GRU_CELL gate arrays. `prev` is a Tensor* owned by
           the caller (typically a per-sequence state handle); we don't
           own it. */
        if (e->op == OP_GRU_CELL && e->op_meta) {
            GruCellMeta* meta = (GruCellMeta*)e->op_meta;
            free(meta->zG); free(meta->rG); free(meta->nG);
            meta->zG = meta->rG = meta->nG = NULL;
            meta->prev = NULL;
        }
        /* Free OP_EMBEDDING indices */
        if (e->op == OP_EMBEDDING && e->op_meta) {
            EmbeddingMeta* meta = (EmbeddingMeta*)e->op_meta;
            free(meta->indices);
            meta->indices = NULL;
        }
        /* Free OP_BATCH_NORM arrays */
        if (e->op == OP_BATCH_NORM && e->op_meta) {
            BatchNormMeta* meta = (BatchNormMeta*)e->op_meta;
            free(meta->x_hat);
            free(meta->rstd);
            meta->x_hat = NULL;
            meta->rstd = NULL;
        }
        /* Free OP_DROPOUT mask */
        if (e->op == OP_DROPOUT && e->op_meta) {
            DropoutMeta* meta = (DropoutMeta*)e->op_meta;
            free(meta->mask);
            meta->mask = NULL;
        }
        /* Free OP_MAX_POOL1D max indices */
        if (e->op == OP_MAX_POOL1D && e->op_meta) {
            MaxPool1DMeta* meta = (MaxPool1DMeta*)e->op_meta;
            free(meta->max_indices);
            meta->max_indices = NULL;
        }
        /* Free OP_MAX_POOL2D max indices */
        if (e->op == OP_MAX_POOL2D && e->op_meta) {
            MaxPool2DMeta* meta = (MaxPool2DMeta*)e->op_meta;
            free(meta->max_indices);
            meta->max_indices = NULL;
        }
        /* Free OP_MAX_POOL2D_BATCHED max indices */
        if (e->op == OP_MAX_POOL2D_BATCHED && e->op_meta) {
            MaxPool2DBatchedMeta* meta = (MaxPool2DBatchedMeta*)e->op_meta;
            free(meta->max_indices);
            meta->max_indices = NULL;
        }
        /* Free grad arrays on non-persistent (arena) tensors.
           These are heap-allocated by ensure_grad during backward. */
        Tensor* r = e->result;
        if (r && !r->persistent && r->grad) {
            free(r->grad);
            r->grad = NULL;
        }
        }
        remaining -= n;
    }
    typed_arena_reset(&tape_arena);
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

/* Refcount API — currently a no-op on the tape backend. The arena lifecycle
 * (tape_reset clears the whole arena at once) doesn't yet participate in
 * per-tensor refcount tracking. Phase 2.4 wires it in. Stubs exist so the
 * multi-link build resolves these symbols across all backends. */
void tensor_retain_handle(TensorHandle h) { (void)h; }
void tensor_release_handle(TensorHandle h) { (void)h; }

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

void tensor_to_floats(TensorHandle h, float* out) {
    Tensor* t = (Tensor*)h;
    for (int i = 0; i < t->numel; i++) out[i] = (float)t->data[i];
}

const char* tensor_dtype_name(TensorHandle h) {
    (void)h;
    /* Tape backend's arena is f64-only — no parallel f32 storage yet
       (see L60 "F32 storage on tape backend" row). Every Tensor* here
       is logically F64 regardless of how the typeclass dispatched
       construction. */
    return "F64";
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

/* ----------------------------------------------------------------
   Numpy-style broadcast helpers for elementwise binary ops.
   Used by binop_elementwise forward and OP_ADD/SUB/MUL/DIV/POW
   backward grad-reduction.
   ---------------------------------------------------------------- */

#define MAX_BCAST_RANK 8

/* True if `a`'s shape exactly matches `r`'s shape (no broadcast). */
static int shapes_equal(Tensor* a, Tensor* r) {
    if (a->numel != r->numel || a->rank != r->rank) return 0;
    for (int k = 0; k < a->rank; k++) {
        if (a->shape[k] != r->shape[k]) return 0;
    }
    return 1;
}

/* Compute broadcast output shape from a and b (right-aligned, numpy rules).
   Returns 1 on success, 0 on incompatible shapes. */
static int compute_bcast_shape(Tensor* a, Tensor* b,
                                int* r_shape, int* r_rank, int* r_numel) {
    int rank = a->rank > b->rank ? a->rank : b->rank;
    if (rank > MAX_BCAST_RANK) return 0;
    int numel = 1;
    for (int k = rank - 1; k >= 0; k--) {
        int ai = k - (rank - a->rank);
        int bi = k - (rank - b->rank);
        int sa = (ai >= 0) ? a->shape[ai] : 1;
        int sb = (bi >= 0) ? b->shape[bi] : 1;
        if (sa != sb && sa != 1 && sb != 1) return 0;
        int so = sa > sb ? sa : sb;
        r_shape[k] = so;
        numel *= so;
    }
    *r_rank = rank;
    *r_numel = numel;
    return 1;
}

/* Compute right-aligned broadcast strides for `a` w.r.t. output shape r_shape.
   out_strides[k] is the increment to a's flat index when output dim k advances;
   0 means broadcast on that dim. r_rank may exceed a->rank (rank-padding). */
static void compute_bcast_strides(Tensor* a, int r_rank, int* r_shape,
                                   int* out_strides) {
    int a_rank = a->rank;
    int natural[MAX_BCAST_RANK];
    int s = 1;
    for (int k = a_rank - 1; k >= 0; k--) { natural[k] = s; s *= a->shape[k]; }
    int offset = r_rank - a_rank;
    for (int j = 0; j < r_rank; j++) {
        int ai = j - offset;
        if (ai < 0) {
            out_strides[j] = 0;  /* phantom dim from rank-padding */
        } else {
            int sa = a->shape[ai];
            int sr = r_shape[j];
            out_strides[j] = (sa == 1 && sr > 1) ? 0 : natural[ai];
        }
    }
}

/* Element-wise binary ops: scalar, same-shape (vDSP fast path), and
   general numpy-style broadcast (e.g. (n,1)*(n,m), (1,m)*(n,m), (m,)*(n,m)). */
static TensorHandle binop_elementwise_inner(TensorHandle ha, TensorHandle hb, int op_tag,
                                       double (*scalar_fn)(double, double));
static TensorHandle binop_elementwise(TensorHandle ha, TensorHandle hb, int op_tag,
                                       double (*scalar_fn)(double, double)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}
static TensorHandle binop_elementwise_inner(TensorHandle ha, TensorHandle hb, int op_tag,
                                       double (*scalar_fn)(double, double)) {
    Tensor* a = (Tensor*)ha; Tensor* b = (Tensor*)hb;
    int rg = a->requires_grad || b->requires_grad;

    /* Both scalar */
    if (a->numel == 1 && b->numel == 1) {
        Tensor* r = make_scalar(scalar_fn(a->data[0], b->data[0]), rg);
        if (rg) tape_append(op_tag, r, a, b, 0);
        return r;
    }

    /* Same shape — vDSP fast path */
    if (a->numel == b->numel && a->rank == b->rank) {
        int same = 1;
        for (int k = 0; k < a->rank; k++) {
            if (a->shape[k] != b->shape[k]) { same = 0; break; }
        }
        if (same) {
            int n = a->numel;
            extern int prof_binop_path_count[];
            prof_binop_path_count[0]++;
            double* data = arena_alloc(n * sizeof(double));
            /* Direct kernel-time probe (diagnostic): measure the actual
               vDSP call separately from the tape_append attribution.
               Compare prof_kernel_per_op vs prof_forward_per_op to see how
               much of the "ADD bucket" is actually kernel work. */
            extern double prof_kernel_per_op[];
            extern int prof_kernel_count_per_op[];
            double _k0 = _wall_ms();
#ifdef __APPLE__
            vDSP_Length vn = (vDSP_Length)n;
            switch (op_tag) {
                case OP_ADD: vDSP_vaddD(a->data, 1, b->data, 1, data, 1, vn); break;
                case OP_SUB: vDSP_vsubD(b->data, 1, a->data, 1, data, 1, vn); break;
                case OP_MUL: vDSP_vmulD(a->data, 1, b->data, 1, data, 1, vn); break;
                case OP_DIV: vDSP_vdivD(b->data, 1, a->data, 1, data, 1, vn); break;
                default:
                    for (int i = 0; i < n; i++) data[i] = scalar_fn(a->data[i], b->data[i]);
                    break;
            }
#else
            for (int i = 0; i < n; i++) data[i] = scalar_fn(a->data[i], b->data[i]);
#endif
            if (op_tag >= 0 && op_tag < OP_COUNT) {
                prof_kernel_per_op[op_tag] += _wall_ms() - _k0;
                prof_kernel_count_per_op[op_tag]++;
            }
            Tensor* r = make_tensor_arena(data, n, a->shape, a->rank, rg);
            if (rg) tape_append(op_tag, r, a, b, 0);
            return r;
        }
    }

    /* Scalar broadcast (one side is rank-0 / numel=1) */
    if (a->numel == 1 || b->numel == 1) {
        Tensor* big = (a->numel == 1) ? b : a;
        double sv = (a->numel == 1) ? a->data[0] : b->data[0];
        int n = big->numel;
        extern int prof_binop_path_count[];
        prof_binop_path_count[1]++;
        double* data = arena_alloc(n * sizeof(double));
        if (a->numel == 1) {
            for (int i = 0; i < n; i++) data[i] = scalar_fn(sv, big->data[i]);
        } else {
            for (int i = 0; i < n; i++) data[i] = scalar_fn(big->data[i], sv);
        }
        Tensor* r = make_tensor_arena(data, n, big->shape, big->rank, rg);
        if (rg) tape_append(op_tag, r, a, b, 0);
        return r;
    }

    /* General broadcast */
    {
        extern int prof_binop_path_count[];
        extern double prof_binop_general_ms;
        prof_binop_path_count[2]++;
        /* Log shapes once per op_tag so we can spot which call sites
           are taking the slow path. */
        static int logged[OP_COUNT] = {0};
        if (op_tag >= 0 && op_tag < OP_COUNT && !logged[op_tag]) {
            logged[op_tag] = 1;
            fprintf(stderr, "[tape diag] op=%d general_bcast shapes: a=[", op_tag);
            for (int k = 0; k < a->rank; k++)
                fprintf(stderr, "%d%s", a->shape[k], k+1<a->rank?",":"");
            fprintf(stderr, "] b=[");
            for (int k = 0; k < b->rank; k++)
                fprintf(stderr, "%d%s", b->shape[k], k+1<b->rank?",":"");
            fprintf(stderr, "]\n");
        }
    }
    double _gb0 = _wall_ms();
    int r_shape[MAX_BCAST_RANK], r_rank, r_numel;
    if (!compute_bcast_shape(a, b, r_shape, &r_rank, &r_numel)) {
        fprintf(stderr, "binop_elementwise: incompatible shapes\n");
        abort();
    }
    int a_strides[MAX_BCAST_RANK], b_strides[MAX_BCAST_RANK];
    compute_bcast_strides(a, r_rank, r_shape, a_strides);
    compute_bcast_strides(b, r_rank, r_shape, b_strides);

    double* data = arena_alloc(r_numel * sizeof(double));
    int idx[MAX_BCAST_RANK] = {0};
    for (int i = 0; i < r_numel; i++) {
        int ai = 0, bi = 0;
        for (int k = 0; k < r_rank; k++) {
            ai += idx[k] * a_strides[k];
            bi += idx[k] * b_strides[k];
        }
        data[i] = scalar_fn(a->data[ai], b->data[bi]);
        for (int k = r_rank - 1; k >= 0; k--) {
            if (++idx[k] < r_shape[k]) break;
            idx[k] = 0;
        }
    }
    Tensor* r = make_tensor_arena(data, r_numel, r_shape, r_rank, rg);
    extern double prof_binop_general_ms;
    prof_binop_general_ms += _wall_ms() - _gb0;
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
    int n = a->numel;
    double* data = arena_alloc(n * sizeof(double));
#ifdef __APPLE__
    {
        int vn = n;
        int used_vdsp = 1;
        switch (op) {
            case OP_NEG: vDSP_vnegD(a->data, 1, data, 1, (vDSP_Length)n); break;
            case OP_EXP: vvexp(data, a->data, &vn); break;
            case OP_LOG: vvlog(data, a->data, &vn); break;
            case OP_SQRT: vvsqrt(data, a->data, &vn); break;
            case OP_TANH: vvtanh(data, a->data, &vn); break;
            case OP_ABS: vvfabs(data, a->data, &vn); break;
            default: used_vdsp = 0; break;
        }
        if (!used_vdsp)
            for (int i = 0; i < n; i++) data[i] = fn(a->data[i]);
    }
#else
    for (int i = 0; i < n; i++) data[i] = fn(a->data[i]);
#endif
    Tensor* r = make_tensor_arena(data, n, a->shape, a->rank, a->requires_grad);
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

/* LeakyReLU: max(alpha*x, x). Uses scalar_arg to store alpha. */
TensorHandle tensor_leaky_relu(TensorHandle ha, double alpha) {
    Tensor* a = (Tensor*)ha;
    if (a->numel == 1) {
        double x = a->data[0];
        Tensor* r = make_scalar(x >= 0 ? x : alpha * x, a->requires_grad);
        if (a->requires_grad) tape_append(OP_LEAKY_RELU, r, a, NULL, alpha);
        return r;
    }
    double* data = malloc(a->numel * sizeof(double));
    for (int i = 0; i < a->numel; i++) {
        double x = a->data[i];
        data[i] = x >= 0 ? x : alpha * x;
    }
    Tensor* r = make_tensor(data, a->shape, a->rank, a->requires_grad);
    free(data);
    if (a->requires_grad) tape_append(OP_LEAKY_RELU, r, a, NULL, alpha);
    return r;
}

/* SiLU / Swish: x * sigmoid(x) */
static double fn_silu(double x) { return x / (1.0 + exp(-x)); }
TensorHandle tensor_silu(TensorHandle a) { return unop_elementwise(a, OP_SILU, fn_silu); }

/* Softplus: log(1 + exp(x)). Numerically stable formulation for large |x|.
 * Backward: f'(x) = 1 / (1 + exp(-x)) = sigmoid(x). */
static double fn_softplus(double x) {
    /* For x > 30, log(1+exp(x)) ≈ x; for x < -30, ≈ exp(x); else direct */
    if (x > 30.0) return x;
    if (x < -30.0) return exp(x);
    return log(1.0 + exp(x));
}
TensorHandle tensor_softplus(TensorHandle a) { return unop_elementwise(a, OP_SOFTPLUS, fn_softplus); }

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

TensorHandle tensor_min(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    double m = t->data[0];
    for (int i = 1; i < t->numel; i++) if (t->data[i] < m) m = t->data[i];
    return make_scalar(m, 0);  /* non-differentiable reduction */
}

TensorHandle tensor_max(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    double m = t->data[0];
    for (int i = 1; i < t->numel; i++) if (t->data[i] > m) m = t->data[i];
    return make_scalar(m, 0);  /* non-differentiable reduction */
}

/* ================================================================
   Linear algebra
   ================================================================ */

TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
    Tensor* mat = (Tensor*)hmat;
    Tensor* vec = (Tensor*)hvec;
    int m = mat->shape[0], n = mat->shape[1];
    int out_shape[] = {m};
    /* Output goes straight into the arena — no calloc/free + memcpy
       round-trip via make_tensor. dgemv's beta=0 means it overwrites
       without reading, so an uninitialized arena buffer is correct. */
    double* out_data = arena_alloc(m * sizeof(double));

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

    Tensor* r = make_tensor_arena(out_data, m, out_shape, 1,
                                  mat->requires_grad || vec->requires_grad);
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MV, r, mat, vec, 0);
        MvMeta* meta = arena_alloc(sizeof(MvMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        memcpy(meta->x_vals, vec->data, n * sizeof(double));
        e->op_meta = meta;
    }
    return r;
}

/* Fused batched linear: Y[B,o] = X[B,i] @ W[o,i]^T + bias[o].
   Single allocation, single tape entry. W: [o, i], X: [B, i], bias: [o] (or NULL). */
TensorHandle tensor_linear_2d(TensorHandle hW, TensorHandle hX, TensorHandle hbias) {
    Tensor* W = (Tensor*)hW;
    Tensor* X = (Tensor*)hX;
    Tensor* bias = (Tensor*)hbias;
    int oo = W->shape[0], ii = W->shape[1];
    int BB = X->shape[0];
    int out_shape[] = {BB, oo};
    double* out_data = malloc(BB * oo * sizeof(double));

    /* Y = X @ W^T : [B,i] @ [i,o] -> [B,o]. Use dgemm with B=W transposed. */
#ifdef __APPLE__
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                BB, oo, ii, 1.0,
                X->data, ii,
                W->data, ii,
                0.0, out_data, oo);
#else
    for (int b = 0; b < BB; b++) {
        for (int o = 0; o < oo; o++) {
            double s = 0;
            for (int j = 0; j < ii; j++) s += X->data[b*ii+j] * W->data[o*ii+j];
            out_data[b*oo+o] = s;
        }
    }
#endif

    /* Y += bias broadcast across batch */
    if (bias) {
        for (int b = 0; b < BB; b++) {
#ifdef __APPLE__
            vDSP_vaddD(out_data + b*oo, 1, bias->data, 1, out_data + b*oo, 1, (vDSP_Length)oo);
#else
            for (int o = 0; o < oo; o++) out_data[b*oo+o] += bias->data[o];
#endif
        }
    }

    int rg = W->requires_grad || X->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor(out_data, out_shape, 2, rg);
    free(out_data);
    if (rg) {
        TapeEntry* e = tape_append(OP_LINEAR_2D, r, W, X, 0);
        Linear2dMeta* meta = arena_alloc(sizeof(Linear2dMeta));
        meta->B = BB; meta->i = ii; meta->o = oo;
        meta->x_vals = arena_alloc(BB * ii * sizeof(double));
        memcpy(meta->x_vals, X->data, BB * ii * sizeof(double));
        meta->bias = bias;
        e->op_meta = meta;
    }
    return r;
}

/* Concatenate along axis 1: A[m,n] ++ B[m,k] -> [m, n+k].
   Single tape entry; backward scatters dY back to dA / dB by column split. */
TensorHandle tensor_concat_2d_axis1(TensorHandle hA, TensorHandle hB) {
    Tensor* A = (Tensor*)hA;
    Tensor* B = (Tensor*)hB;
    int m = A->shape[0];
    int n = A->shape[1];
    int k = B->shape[1];
    int out_shape[] = {m, n + k};
    double* out_data = malloc(m * (n + k) * sizeof(double));
    for (int i = 0; i < m; i++) {
        memcpy(out_data + i * (n + k), A->data + i * n, n * sizeof(double));
        memcpy(out_data + i * (n + k) + n, B->data + i * k, k * sizeof(double));
    }
    int rg = A->requires_grad || B->requires_grad;
    Tensor* r = make_tensor(out_data, out_shape, 2, rg);
    free(out_data);
    if (rg) tape_append(OP_CONCAT_2D_AXIS1, r, A, B, (double)n);
    return r;
}

/* Fused linear: y = W @ x + bias. Single allocation, single tape entry.
   W: [m, n], x: [n], bias: [m] (or NULL). Result: [m]. */
TensorHandle tensor_linear(TensorHandle hW, TensorHandle hx, TensorHandle hbias) {
    Tensor* W = (Tensor*)hW;
    Tensor* x = (Tensor*)hx;
    Tensor* bias = (Tensor*)hbias;
    int m = W->shape[0], n = W->shape[1];
    int out_shape[] = {m};
    /* Arena alloc — same fast path as tensor_mv. */
    double* out_data = arena_alloc(m * sizeof(double));

    /* y = W @ x */
#ifdef __APPLE__
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
                W->data, n, x->data, 1, 0.0, out_data, 1);
#else
    for (int i = 0; i < m; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += W->data[i*n+j] * x->data[j];
        out_data[i] = s;
    }
#endif

    /* y += bias */
    if (bias) {
#ifdef __APPLE__
        vDSP_vaddD(out_data, 1, bias->data, 1, out_data, 1, (vDSP_Length)m);
#else
        for (int i = 0; i < m; i++) out_data[i] += bias->data[i];
#endif
    }

    int rg = W->requires_grad || x->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor_arena(out_data, m, out_shape, 1, rg);
    if (rg) {
        TapeEntry* e = tape_append(OP_LINEAR, r, W, x, 0);
        LinearMeta* meta = arena_alloc(sizeof(LinearMeta));
        meta->m = m; meta->n = n;
        meta->x_vals = arena_alloc(n * sizeof(double));
        memcpy(meta->x_vals, x->data, n * sizeof(double));
        meta->bias = bias;
        e->op_meta = meta;
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

TensorHandle tensor_reshape_4d(TensorHandle h, int d0, int d1, int d2, int d3) {
    int shape[] = {d0, d1, d2, d3};
    return tensor_reshape(h, shape, 4);
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

typedef struct { int m, n, rep0, rep1; } Tile2dMeta;

TensorHandle tensor_tile_2d(TensorHandle h, int rep0, int rep1) {
    Tensor* t = (Tensor*)h;
    int m = t->shape[0], n = t->shape[1];
    int M = m * rep0, N = n * rep1;
    double* data = malloc(M * N * sizeof(double));
    /* output[i, j] = input[i mod m, j mod n] (tile semantics) */
    for (int i = 0; i < M; i++) {
        int si = i % m;
        for (int j = 0; j < N; j++) {
            data[i * N + j] = t->data[si * n + (j % n)];
        }
    }
    int shape[] = {M, N};
    Tensor* r = make_tensor(data, shape, 2, t->requires_grad);
    free(data);
    if (t->requires_grad) {
        Tile2dMeta* meta = arena_alloc(sizeof(Tile2dMeta));
        meta->m = m; meta->n = n; meta->rep0 = rep0; meta->rep1 = rep1;
        TapeEntry* e = tape_append(OP_TILE_2D, r, t, NULL, 0);
        e->op_meta = meta;
    }
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
        TapeEntry* e = tape_append(OP_LAYER_NORM_2D, r, t, NULL, 0);
        e->op_meta = meta;
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
        TapeEntry* e = tape_append(OP_SOFTMAX, r, t, NULL, 0);
        SoftmaxMeta* meta = arena_alloc(sizeof(SoftmaxMeta));
        meta->n = n;
        meta->out_vals = r->data;  /* r persists in arena — safe to reference */
        e->op_meta = meta;
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
   Cross-Attention: Q @ K^T * scale [+ mask] -> softmax -> @ V
   Q [B,seqQ,d], K [B,seqK,d], V [B,seqK,d] -> [B,seqQ,d]
   ================================================================ */

TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                    TensorHandle hmask, double scale) {
    /* Compose from existing ops — backward handled by tape of individual ops */
    TensorHandle KT = tensor_transpose_last2(hK);
    TensorHandle scores = tensor_mul_scalar(tensor_bmm_3x3(hQ, KT), scale);
    if (hmask) scores = tensor_masked_fill(scores, hmask, -1.0e20);
    TensorHandle attn = tensor_softmax_3d(scores);
    return tensor_bmm_3x3(attn, hV);
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
        TapeEntry* e = tape_append(OP_EMBEDDING, r, weight, NULL, 0);
        EmbeddingMeta* meta = arena_alloc(sizeof(EmbeddingMeta));
        meta->n = n;
        meta->embedDim = embedDim;
        meta->indices = idx_copy;
        e->op_meta = meta;
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
        TapeEntry* e = tape_append(OP_BATCH_NORM, r, input, NULL, 0);
        BatchNormMeta* meta = arena_alloc(sizeof(BatchNormMeta));
        meta->gamma = gamma;
        meta->beta = beta;
        meta->x_hat = x_hat;
        meta->rstd = rstd;
        meta->C = C;
        meta->spatial = spatial;
        e->op_meta = meta;
    } else {
        free(x_hat);
        free(rstd);
    }
    return r;
}

/* ================================================================
   Group Normalization: normalize within channel groups
   ================================================================ */

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    Tensor* input = (Tensor*)hinput;
    Tensor* gamma = (Tensor*)hgamma;
    Tensor* beta = (Tensor*)hbeta;
    int n = channels * spatial;
    int chPerGroup = channels / numGroups;
    int groupSize = chPerGroup * spatial;

    double* out = calloc(n, sizeof(double));
    for (int g = 0; g < numGroups; g++) {
        /* Compute mean and var for this group */
        double mean = 0;
        int base = g * groupSize;
        for (int j = 0; j < groupSize; j++) mean += input->data[base + j];
        mean /= groupSize;
        double var = 0;
        for (int j = 0; j < groupSize; j++) {
            double d = input->data[base + j] - mean;
            var += d * d;
        }
        var /= groupSize;
        double rstd = 1.0 / sqrt(var + eps);
        /* Normalize, then scale+shift per-channel */
        for (int c = 0; c < chPerGroup; c++) {
            int absC = g * chPerGroup + c;
            for (int s = 0; s < spatial; s++) {
                int idx = absC * spatial + s;
                double x_hat = (input->data[idx] - mean) * rstd;
                out[idx] = gamma->data[absC] * x_hat + beta->data[absC];
            }
        }
    }
    int out_shape[] = {n};
    Tensor* r = make_tensor(out, out_shape, 1, input->requires_grad || gamma->requires_grad);
    free(out);
    /* No backward tape entry for now — torch/MLX handle it natively */
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
        TapeEntry* e = tape_append(OP_DROPOUT, r, input, NULL, 0);
        DropoutMeta* meta = arena_alloc(sizeof(DropoutMeta));
        meta->mask = mask;
        meta->numel = n;
        e->op_meta = meta;
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
    /* Record tape entry: backward scatters grad back to input positions.
       index stored as arg2 (non-grad integer tensor). */
    if (r->requires_grad) {
        tape_append(OP_GATHER, r, input, index, (double)n);
    }
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
    /* Record tape entry: backward gathers grad back to src positions.
       index stored as arg2 (non-grad integer tensor), src as arg1. */
    if (r->requires_grad) {
        tape_append(OP_SCATTER_ADD, r, src, index, (double)out_size);
    }
    return r;
}

/* ================================================================
   Sort / Scan
   ================================================================ */

/* Comparison for ascending argsort */
static const double* argsort_data_ptr;
static int argsort_cmp_asc(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    double da = argsort_data_ptr[ia], db = argsort_data_ptr[ib];
    return (da > db) - (da < db);
}
static int argsort_cmp_desc(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    double da = argsort_data_ptr[ia], db = argsort_data_ptr[ib];
    return (db > da) - (db < da);
}

TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
    (void)dim; /* only 1D supported */
    Tensor* t = (Tensor*)ht;
    int n = t->numel;
    int* indices = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;
    argsort_data_ptr = t->data;
    qsort(indices, n, sizeof(int), descending ? argsort_cmp_desc : argsort_cmp_asc);
    double* out = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) out[i] = (double)indices[i];
    free(indices);
    int shape[] = {n};
    Tensor* r = make_tensor(out, shape, 1, 0); /* integer indices: no grad */
    free(out);
    return r;
}

TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    (void)dim; /* only 1D supported */
    Tensor* t = (Tensor*)ht;
    int n = t->numel;
    double* out = malloc(n * sizeof(double));
    double prod = 1.0;
    for (int i = 0; i < n; i++) {
        prod *= t->data[i];
        out[i] = prod;
    }
    int shape[] = {n};
    Tensor* r = make_tensor(out, shape, 1, t->requires_grad);
    free(out);
    if (r->requires_grad) {
        tape_append(OP_CUMPROD, r, t, NULL, 0);
    }
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
        TapeEntry* e = tape_append(OP_AVG_POOL1D, r, input, NULL, 0);
        AvgPool1DMeta* meta = arena_alloc(sizeof(AvgPool1DMeta));
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        e->op_meta = meta;
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
        TapeEntry* e = tape_append(OP_AVG_POOL2D, r, input, NULL, 0);
        AvgPool2DMeta* meta = arena_alloc(sizeof(AvgPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW; meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
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
        TapeEntry* e = tape_append(OP_CONV1D, r, input, kernel, 0);
        Conv1DMeta* meta = arena_alloc(sizeof(Conv1DMeta));
        meta->inC = inC; meta->outC = outC; meta->L = L;
        meta->kL = kL; meta->pad = pad; meta->stride = stride; meta->oL = oL;
        e->op_meta = meta;
        e->inputs = (Tensor**)bias;
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
        TapeEntry* e = tape_append(OP_MAX_POOL1D, r, input, NULL, 0);
        MaxPool1DMeta* meta = arena_alloc(sizeof(MaxPool1DMeta));
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        meta->max_indices = max_idx;
        e->op_meta = meta;
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
   Transposed Convolution
   ConvTranspose1D: output[oc][ol] = sum over ic,kl of input[ic][il] * kernel[ic][oc][kl]
   where ol = il*stride - pad + kl
   ================================================================ */

TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int pad, int stride) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[1], kL = kernel->shape[2];
    int oL = (L - 1) * stride - 2 * pad + kL;
    double* out = calloc(outC * oL, sizeof(double));
    if (bias) for (int oc = 0; oc < outC; oc++)
        for (int ol = 0; ol < oL; ol++) out[oc*oL + ol] = bias->data[oc];
    for (int ic = 0; ic < inC; ic++)
        for (int il = 0; il < L; il++)
            for (int oc = 0; oc < outC; oc++)
                for (int kl = 0; kl < kL; kl++) {
                    int ol = il * stride - pad + kl;
                    if (ol >= 0 && ol < oL)
                        out[oc*oL + ol] += input->data[ic*L + il] * kernel->data[ic*outC*kL + oc*kL + kl];
                }
    int out_shape[] = {outC, oL};
    Tensor* r = make_tensor(out, out_shape, 2, input->requires_grad || kernel->requires_grad);
    free(out);
    return r;
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int padH, int padW,
                                     int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[1], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H - 1) * strideH - 2 * padH + kH;
    int oW = (W - 1) * strideW - 2 * padW + kW;
    double* out = calloc(outC * oH * oW, sizeof(double));
    if (bias) for (int oc = 0; oc < outC; oc++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) out[oc*oH*oW + oh*oW + ow] = bias->data[oc];
    for (int ic = 0; ic < inC; ic++)
        for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
                for (int oc = 0; oc < outC; oc++)
                    for (int kh = 0; kh < kH; kh++)
                        for (int kw = 0; kw < kW; kw++) {
                            int oh = ih*strideH - padH + kh;
                            int ow = iw*strideW - padW + kw;
                            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
                                out[oc*oH*oW + oh*oW + ow] += input->data[ic*H*W + ih*W + iw]
                                    * kernel->data[ic*outC*kH*kW + oc*kH*kW + kh*kW + kw];
                        }
    int out_shape[] = {outC, oH, oW};
    Tensor* r = make_tensor(out, out_shape, 3, input->requires_grad || kernel->requires_grad);
    free(out);
    return r;
}

/* ================================================================
   Grouped Convolution: splits channels into groups, applies separate convs.
   For tape backend, we just call the ungrouped conv per-group.
   ================================================================ */

TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int pad, int stride, int groups) {
    if (groups == 1) return tensor_conv1d(hinput, hkernel, hbias, pad, stride);
    /* Decompose into per-group conv1d and concatenate */
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[0];
    int inC_g = inC / groups;
    int outC_g = outC / groups;
    int kL = kernel->shape[2];
    int oL = (L + 2*pad - kL) / stride + 1;
    int total = outC * oL;
    double* out = calloc(total, sizeof(double));
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < outC_g; oc++) {
            int abs_oc = g * outC_g + oc;
            for (int ol = 0; ol < oL; ol++) {
                double val = bias ? bias->data[abs_oc] : 0.0;
                for (int ic = 0; ic < inC_g; ic++) {
                    int abs_ic = g * inC_g + ic;
                    for (int kl = 0; kl < kL; kl++) {
                        int il = ol * stride - pad + kl;
                        if (il >= 0 && il < L)
                            val += input->data[abs_ic*L + il] * kernel->data[abs_oc*inC_g*kL + ic*kL + kl];
                    }
                }
                out[abs_oc*oL + ol] = val;
            }
        }
    }
    int out_shape[] = {outC, oL};
    Tensor* r = make_tensor(out, out_shape, 2, input->requires_grad || kernel->requires_grad);
    free(out);
    /* No separate backward for grouped — reuse OP_CONV1D with groups=1 per group.
       For simplicity, grouped conv on tape backend doesn't support backward yet.
       Torch and MLX backends use native grouped conv with full autograd. */
    return r;
}

TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int padH, int padW,
                                   int strideH, int strideW, int groups) {
    if (groups == 1) return tensor_conv2d(hinput, hkernel, hbias, padH, padW, strideH, strideW);
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[0];
    int inC_g = inC / groups;
    int outC_g = outC / groups;
    int kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    double* out = calloc(outC * oH * oW, sizeof(double));
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < outC_g; oc++) {
            int abs_oc = g * outC_g + oc;
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double val = bias ? bias->data[abs_oc] : 0.0;
                    for (int ic = 0; ic < inC_g; ic++) {
                        int abs_ic = g * inC_g + ic;
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh*strideH - padH + kh;
                                int iw = ow*strideW - padW + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    val += input->data[abs_ic*H*W + ih*W + iw]
                                         * kernel->data[abs_oc*inC_g*kH*kW + ic*kH*kW + kh*kW + kw];
                            }
                    }
                    out[abs_oc*oH*oW + oh*oW + ow] = val;
                }
        }
    }
    int out_shape[] = {outC, oH, oW};
    Tensor* r = make_tensor(out, out_shape, 3, input->requires_grad || kernel->requires_grad);
    free(out);
    return r;
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
        TapeEntry* e = tape_append(OP_CONV2D, r, input, kernel, 0);
        Conv2DMeta* meta = arena_alloc(sizeof(Conv2DMeta));
        meta->inC = inC; meta->outC = outC;
        meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
        /* Store bias pointer in scalar_arg slot (cast) for backward */
        e->inputs = (Tensor**)bias;  /* reuse inputs field for bias ptr */
    }
    return r;
}

/* ================================================================
   Batched Conv2D: input [B, inC, H, W] x kernel [outC, inC, kH, kW]
                   + bias [outC] -> [B, outC, oH, oW]

   Forward uses the standard im2col + cblas_dgemm decomposition:
       X_col [M, K] where M = B*oH*oW, K = inC*kH*kW
       Y_unf [M, outC] = X_col @ W^T   (single dgemm)
       out   [B, outC, oH, oW] = permute(Y_unf, (0,2,1) on (B*oHoW, outC))
                                + bias broadcast.
   This is what PyTorch / cuDNN do at the unfused-conv path; the dgemm
   replaces an O(B·outC·inC·kH·kW·oH·oW) hand-rolled triple loop with
   Apple Accelerate's blocked sgemm.
   ================================================================ */

/* im2col: build X_col [M, K] where M = B*oH*oW, K = inC*kH*kW.
   Each row is one (batch, out-position)'s unfolded inC*kH*kW window. */
static void conv2d_im2col(const double* input, int B, int inC, int H, int W,
                          int kH, int kW, int padH, int padW,
                          int strideH, int strideW, int oH, int oW,
                          double* X_col) {
    int K = inC * kH * kW;
    int M = B * oH * oW;
    memset(X_col, 0, (size_t)M * K * sizeof(double));
    for (int b = 0; b < B; b++) {
        const double* inp_b = input + (size_t)b * inC * H * W;
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double* row = X_col + ((size_t)b * oH * oW + oh * oW + ow) * K;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * strideH - padH + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * strideW - padW + kw;
                            if (iw < 0 || iw >= W) continue;
                            row[ic * kH * kW + kh * kW + kw] =
                                inp_b[ic * H * W + ih * W + iw];
                        }
                    }
                }
            }
        }
    }
}

/* col2im (gradient accumulating version): scatter dX_col [M, K] back into
   dInput [B, inC, H, W]. Padding cells are dropped. */
static void conv2d_col2im_accumulate(const double* dX_col, int B, int inC,
                                      int H, int W, int kH, int kW,
                                      int padH, int padW, int strideH,
                                      int strideW, int oH, int oW,
                                      double* dInput) {
    int K = inC * kH * kW;
    for (int b = 0; b < B; b++) {
        double* din_b = dInput + (size_t)b * inC * H * W;
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                const double* row = dX_col + ((size_t)b * oH * oW + oh * oW + ow) * K;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * strideH - padH + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * strideW - padW + kw;
                            if (iw < 0 || iw >= W) continue;
                            din_b[ic * H * W + ih * W + iw] +=
                                row[ic * kH * kW + kh * kW + kw];
                        }
                    }
                }
            }
        }
    }
}

TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                    TensorHandle hbias, int padH, int padW,
                                    int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;

    int B = input->shape[0], inC = input->shape[1];
    int H = input->shape[2], W = input->shape[3];
    int outC = kernel->shape[0], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int K = inC * kH * kW;
    int M = B * oH * oW;
    int out_numel = B * outC * oH * oW;

    /* Build X_col [M, K] — local workspace (heap, freed at end). */
    double* X_col = (double*)calloc((size_t)M * K, sizeof(double));
    conv2d_im2col(input->data, B, inC, H, W, kH, kW, padH, padW,
                  strideH, strideW, oH, oW, X_col);

    /* Y_unf [M, outC] = X_col [M, K] @ W^T [K, outC]
       W is stored row-major as [outC, K]. */
    double* Y_unf = calloc((size_t)M * outC, sizeof(double));
#ifdef __APPLE__
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, outC, K, 1.0,
                X_col, K,
                kernel->data, K,
                0.0, Y_unf, outC);
#else
    for (int m = 0; m < M; m++)
        for (int oc = 0; oc < outC; oc++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += X_col[m*K + k] * kernel->data[oc*K + k];
            Y_unf[m*outC + oc] = s;
        }
#endif

    /* Permute Y_unf [B*oH*oW, outC] -> out [B, outC, oH, oW] + bias broadcast */
    double* out = calloc(out_numel, sizeof(double));
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < outC; oc++) {
            double b_val = bias ? bias->data[oc] : 0.0;
            double* out_chan = out + ((size_t)b * outC + oc) * oH * oW;
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    int row = b * oH * oW + oh * oW + ow;
                    out_chan[oh*oW + ow] = Y_unf[row * outC + oc] + b_val;
                }
            }
        }
    }
    free(Y_unf);
    free(X_col);

    int out_shape[] = {B, outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);
    Tensor* r = make_tensor(out, out_shape, 4, rg);
    free(out);

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_CONV2D_BATCHED, r, input, kernel, 0);
        Conv2DBatchedMeta* meta = arena_alloc(sizeof(Conv2DBatchedMeta));
        meta->B = B; meta->inC = inC; meta->outC = outC;
        meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
        /* Store bias pointer in scalar_arg slot (cast) for backward */
        e->inputs = (Tensor**)bias;  /* reuse inputs field for bias ptr */
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
        TapeEntry* e = tape_append(OP_MAX_POOL2D, r, input, NULL, 0);
        MaxPool2DMeta* meta = arena_alloc(sizeof(MaxPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;  /* heap-allocated, freed in tape_reset */
        e->op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

/* ================================================================
   Batched MaxPool2D: input [B, C, H, W] -> [B, C, oH, oW]
   ================================================================ */

TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW,
                                        int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int B = input->shape[0], C = input->shape[1];
    int H = input->shape[2], W = input->shape[3];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    int out_per_sample = C * oH * oW;
    int out_numel = B * out_per_sample;

    double* out = calloc(out_numel, sizeof(double));
    int* max_idx = malloc(out_numel * sizeof(int));

    for (int b = 0; b < B; b++) {
        const double* inp_b = input->data + b * C * H * W;
        double* out_b = out + b * out_per_sample;
        int* idx_b = max_idx + b * out_per_sample;
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
                            if (inp_b[flat] > best) {
                                best = inp_b[flat];
                                best_idx = b * C*H*W + flat;  /* absolute index into input.data */
                            }
                        }
                    }
                    int out_idx = c*oH*oW + oh*oW + ow;
                    out_b[out_idx] = best;
                    idx_b[out_idx] = best_idx;
                }
            }
        }
    }

    int out_shape[] = {B, C, oH, oW};
    Tensor* r = make_tensor(out, out_shape, 4, input->requires_grad);
    free(out);

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MAX_POOL2D_BATCHED, r, input, NULL, 0);
        MaxPool2DBatchedMeta* meta = arena_alloc(sizeof(MaxPool2DBatchedMeta));
        meta->B = B; meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;
        e->op_meta = meta;
    } else {
        free(max_idx);
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
static double prof_epoch_start = 0; /* set by backend_epoch_begin() */
static int prof_backward_processed = 0, prof_backward_skipped = 0;
static double prof_backward_per_op[OP_COUNT] = {0};
static int prof_backward_count_per_op[OP_COUNT] = {0};
/* Per-op forward timing. Non-static so the forward declarations near
   tape_append (which is defined earlier in the file) can refer to them. */
double prof_forward_per_op[OP_COUNT] = {0};
int prof_forward_count_per_op[OP_COUNT] = {0};
/* Direct kernel-only timer — only OP_ADD/SUB/MUL/DIV's vDSP path
   populates it today, for diagnosing the "ADD bucket dominates
   forward" attribution. Compare prof_kernel_per_op[OP_ADD] vs
   prof_forward_per_op[OP_ADD] to see how much of the bucket is
   actual kernel time versus inter-op leakage. */
double prof_kernel_per_op[OP_COUNT] = {0};
int prof_kernel_count_per_op[OP_COUNT] = {0};
/* Path-classification counters for binop_elementwise — [fast vDSP,
   scalar bcast, general bcast]. Diagnostic for the
   "ADD bucket dominates forward" investigation. */
int prof_binop_path_count[3] = {0};
double prof_binop_general_ms = 0;
/* Full-function (entry-to-exit) timer for binop_elementwise. Compare
   with prof_forward_per_op to see how much of the attributed bucket
   is actually inside our function vs leaked from somewhere else. */
double prof_binop_inside_ms[OP_COUNT] = {0};
int prof_binop_inside_count[OP_COUNT] = {0};
/* Wall-time of the previous tape_append (or epoch_begin). The delta
   from that moment to the next tape_append is attributed to the op
   being recorded now — i.e. its compute + tape-append cost. Set to 0
   to disable accumulation (e.g. during backward). */
double prof_op_t_prev = 0;



void tensor_backward(TensorHandle h) {
    double t0 = _wall_ms();
    /* Attribute time since epoch_begin to forward */
    if (prof_epoch_start > 0) {
        prof_forward_ms += t0 - prof_epoch_start;
        prof_epoch_start = 0;
    }
    /* Stop per-op forward accumulation; the next epoch_begin will rearm. */
    prof_op_t_prev = 0;
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) return;

    /* Initialize loss gradient to 1.0 */
    ensure_grad(loss);
    loss->grad[0] = 1.0;

    int processed = 0, skipped = 0;

    /* Walk tape in reverse via chunk-array — same semantics as the old
       `for (int i = loss->tape_idx; i >= 0; i--) { TapeEntry* e = &tape[i]; }`
       but indexes the chunked tape directly so the cost stays O(N) total. */
    int _num_chunks_b = 0;
    for (TypedArenaChunk* _c = tape_arena.head; _c; _c = _c->next) _num_chunks_b++;
    TypedArenaChunk** _chunks_b = malloc(_num_chunks_b * sizeof(TypedArenaChunk*));
    { int _ci = 0; for (TypedArenaChunk* _c = tape_arena.head; _c; _c = _c->next) _chunks_b[_ci++] = _c; }
    int _start_cidx = loss->tape_idx / TAPE_CHUNK_SIZE;
    int _start_intra = loss->tape_idx % TAPE_CHUNK_SIZE;
    for (int _cidx = _start_cidx; _cidx >= 0; _cidx--) {
        TapeEntry* _entries_b = (TapeEntry*)_chunks_b[_cidx]->data;
        int _last_intra = (_cidx == _start_cidx) ? _start_intra : TAPE_CHUNK_SIZE - 1;
        for (int _j = _last_intra; _j >= 0; _j--) {
        TapeEntry* e = &_entries_b[_j];
        Tensor* r = e->result;
        if (!r->grad) { skipped++; continue; }
        processed++;
        double t_op = _wall_ms();

        Tensor* a = e->arg1;
        Tensor* b = e->arg2;

        switch (e->op) {
        case OP_CONST: break; /* leaf — grad already accumulated */

        /* Elementwise-binop backward (OP_ADD/SUB/MUL/DIV/POW) — handle three
           cases per side: same-shape (fast loop), scalar (sum-reduce), and
           general numpy-style broadcast (walk r-positions with broadcast
           strides, accumulating into the operand's flat index). */
        case OP_ADD: {
            int a_match = a && shapes_equal(a, r);
            int b_match = b && shapes_equal(b, r);
            if (a) ensure_grad(a);
            if (b) ensure_grad(b);
            ensure_grad(r);
            if (a_match) {
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j];
            } else if (a && a->numel == 1) {
                double s = 0; for (int j = 0; j < r->numel; j++) s += r->grad[j];
                a->grad[0] += s;
            }
            if (b_match) {
                for (int j = 0; j < b->numel; j++) b->grad[j] += r->grad[j];
            } else if (b && b->numel == 1) {
                double s = 0; for (int j = 0; j < r->numel; j++) s += r->grad[j];
                b->grad[0] += s;
            }
            if ((a && !a_match && a->numel != 1) || (b && !b_match && b->numel != 1)) {
                int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
                int idx[MAX_BCAST_RANK] = {0};
                if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
                if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
                int do_a = a && !a_match && a->numel != 1;
                int do_b = b && !b_match && b->numel != 1;
                for (int i = 0; i < r->numel; i++) {
                    if (do_a) {
                        int ai = 0;
                        for (int k = 0; k < r->rank; k++) ai += idx[k] * a_str[k];
                        a->grad[ai] += r->grad[i];
                    }
                    if (do_b) {
                        int bi = 0;
                        for (int k = 0; k < r->rank; k++) bi += idx[k] * b_str[k];
                        b->grad[bi] += r->grad[i];
                    }
                    for (int k = r->rank - 1; k >= 0; k--) {
                        if (++idx[k] < r->shape[k]) break; idx[k] = 0;
                    }
                }
            }
            break;
        }

        case OP_SUB: {
            int a_match = a && shapes_equal(a, r);
            int b_match = b && shapes_equal(b, r);
            if (a) ensure_grad(a);
            if (b) ensure_grad(b);
            ensure_grad(r);
            if (a_match) {
                for (int j = 0; j < a->numel; j++) a->grad[j] += r->grad[j];
            } else if (a && a->numel == 1) {
                double s = 0; for (int j = 0; j < r->numel; j++) s += r->grad[j];
                a->grad[0] += s;
            }
            if (b_match) {
                for (int j = 0; j < b->numel; j++) b->grad[j] -= r->grad[j];
            } else if (b && b->numel == 1) {
                double s = 0; for (int j = 0; j < r->numel; j++) s += r->grad[j];
                b->grad[0] -= s;
            }
            if ((a && !a_match && a->numel != 1) || (b && !b_match && b->numel != 1)) {
                int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
                int idx[MAX_BCAST_RANK] = {0};
                if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
                if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
                int do_a = a && !a_match && a->numel != 1;
                int do_b = b && !b_match && b->numel != 1;
                for (int i = 0; i < r->numel; i++) {
                    if (do_a) {
                        int ai = 0;
                        for (int k = 0; k < r->rank; k++) ai += idx[k] * a_str[k];
                        a->grad[ai] += r->grad[i];
                    }
                    if (do_b) {
                        int bi = 0;
                        for (int k = 0; k < r->rank; k++) bi += idx[k] * b_str[k];
                        b->grad[bi] -= r->grad[i];
                    }
                    for (int k = r->rank - 1; k >= 0; k--) {
                        if (++idx[k] < r->shape[k]) break; idx[k] = 0;
                    }
                }
            }
            break;
        }

        case OP_MUL: {
            int a_match = a && shapes_equal(a, r);
            int b_match = b && shapes_equal(b, r);
            if (a) ensure_grad(a);
            if (b) ensure_grad(b);
            ensure_grad(r);
            /* Fast path: both shapes match r */
            if (a_match && b_match) {
                for (int j = 0; j < r->numel; j++) {
                    a->grad[j] += r->grad[j] * b->data[j];
                    b->grad[j] += r->grad[j] * a->data[j];
                }
            } else {
                /* Mixed: scalar / broadcast on either side. Walk r positions. */
                int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
                int idx[MAX_BCAST_RANK] = {0};
                if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
                if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
                for (int i = 0; i < r->numel; i++) {
                    int ai = 0, bi = 0;
                    for (int k = 0; k < r->rank; k++) {
                        ai += idx[k] * a_str[k];
                        bi += idx[k] * b_str[k];
                    }
                    if (a) a->grad[ai] += r->grad[i] * b->data[bi];
                    if (b) b->grad[bi] += r->grad[i] * a->data[ai];
                    for (int k = r->rank - 1; k >= 0; k--) {
                        if (++idx[k] < r->shape[k]) break; idx[k] = 0;
                    }
                }
            }
            break;
        }

        case OP_DIV: {
            int a_match = a && shapes_equal(a, r);
            int b_match = b && shapes_equal(b, r);
            if (a) ensure_grad(a);
            if (b) ensure_grad(b);
            ensure_grad(r);
            if (a_match && b_match) {
                for (int j = 0; j < r->numel; j++) {
                    double bv = b->data[j];
                    a->grad[j] += r->grad[j] / bv;
                    b->grad[j] -= r->grad[j] * a->data[j] / (bv * bv);
                }
            } else {
                int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
                int idx[MAX_BCAST_RANK] = {0};
                if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
                if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
                for (int i = 0; i < r->numel; i++) {
                    int ai = 0, bi = 0;
                    for (int k = 0; k < r->rank; k++) {
                        ai += idx[k] * a_str[k];
                        bi += idx[k] * b_str[k];
                    }
                    double bv = b->data[bi];
                    if (a) a->grad[ai] += r->grad[i] / bv;
                    if (b) b->grad[bi] -= r->grad[i] * a->data[ai] / (bv * bv);
                    for (int k = r->rank - 1; k >= 0; k--) {
                        if (++idx[k] < r->shape[k]) break; idx[k] = 0;
                    }
                }
            }
            break;
        }

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

        case OP_POW: {
            int a_match = a && shapes_equal(a, r);
            int b_match = b && shapes_equal(b, r);
            if (a) ensure_grad(a);
            if (b) ensure_grad(b);
            ensure_grad(r);
            if (a_match && b_match) {
                for (int j = 0; j < r->numel; j++) {
                    double av = fmax(a->data[j], 1e-20);
                    double bv = b->data[j];
                    a->grad[j] += r->grad[j] * bv * pow(av, bv - 1.0);
                    b->grad[j] += r->grad[j] * r->data[j] * log(av);
                }
            } else {
                int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
                int idx[MAX_BCAST_RANK] = {0};
                if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
                if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
                for (int i = 0; i < r->numel; i++) {
                    int ai = 0, bi = 0;
                    for (int k = 0; k < r->rank; k++) {
                        ai += idx[k] * a_str[k];
                        bi += idx[k] * b_str[k];
                    }
                    double av = fmax(a->data[ai], 1e-20);
                    double bv = b->data[bi];
                    if (a) a->grad[ai] += r->grad[i] * bv * pow(av, bv - 1.0);
                    if (b) b->grad[bi] += r->grad[i] * r->data[i] * log(av);
                    for (int k = r->rank - 1; k >= 0; k--) {
                        if (++idx[k] < r->shape[k]) break; idx[k] = 0;
                    }
                }
            }
            break;
        }

        case OP_SIGMOID: {
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) { double s = r->data[j]; a->grad[j] += r->grad[j] * s * (1.0 - s); } }
            break;
        }

        case OP_TANH: {
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) { double t = r->data[j]; a->grad[j] += r->grad[j] * (1.0 - t * t); } }
            break;
        }

        case OP_SOFTPLUS: {
            /* d/dx softplus(x) = 1 / (1 + exp(-x)) = sigmoid(x). a->data is x. */
            if (a) { ensure_grad(a); for (int j = 0; j < a->numel; j++) { double s = 1.0 / (1.0 + exp(-a->data[j])); a->grad[j] += r->grad[j] * s; } }
            break;
        }

        case OP_TILE_2D: {
            /* Forward: output[i, j] = input[i mod m, j mod n]
               Backward: grad to input[si, sj] = sum over r0, c0 of
                         grad_output[r0*m + si, c0*n + sj] */
            if (a) {
                ensure_grad(a);
                Tile2dMeta* meta = (Tile2dMeta*)e->op_meta;
                int m = meta->m, n = meta->n;
                int rep0 = meta->rep0, rep1 = meta->rep1;
                int N = n * rep1;
                for (int si = 0; si < m; si++) {
                    for (int sj = 0; sj < n; sj++) {
                        double s = 0.0;
                        for (int r0 = 0; r0 < rep0; r0++) {
                            for (int c0 = 0; c0 < rep1; c0++) {
                                s += r->grad[(r0 * m + si) * N + (c0 * n + sj)];
                            }
                        }
                        a->grad[si * n + sj] += s;
                    }
                }
            }
            break;
        }

        case OP_GRU_CELL: {
            /* nn.GRU backward. arg1 = ih, arg2 = hh, prev in meta.
                 z = sigmoid(ih_z + hh_z),  r = sigmoid(ih_r + hh_r)
                 n = tanh(ih_n + r * hh_n)
                 h' = (1-z) * n + z * prev
               Gradient flows:
                 d_z       = dh' * (prev - n)
                 d_z_raw   = d_z * z * (1-z)        (goes to ih_z and hh_z)
                 d_n       = dh' * (1-z)
                 d_n_pre   = d_n * (1-n*n)          (where n_pre = ih_n + r*hh_n)
                 d_ih_n    = d_n_pre
                 d_(r*hh_n)= d_n_pre   →  d_r = d_n_pre * hh_n
                                          d_hh_n = d_n_pre * r
                 d_r_raw   = d_r * r * (1-r)        (goes to ih_r and hh_r)
                 d_prev    = dh' * z                                              */
            GruCellMeta* meta = (GruCellMeta*)e->op_meta;
            int oo = meta->o;
            Tensor* ih = a;
            Tensor* hh = b;
            Tensor* prev = meta->prev;
            ensure_grad(r);
            for (int i = 0; i < oo; i++) {
                double dh = r->grad[i];
                double zv = meta->zG[i];
                double rv = meta->rG[i];
                double nv = meta->nG[i];
                double hh_n_i = hh->data[2*oo + i];

                double d_z_raw = dh * (prev->data[i] - nv) * zv * (1.0 - zv);
                double d_n_pre = dh * (1.0 - zv) * (1.0 - nv * nv);
                double d_r     = d_n_pre * hh_n_i;
                double d_r_raw = d_r * rv * (1.0 - rv);
                double d_hh_n  = d_n_pre * rv;

                if (ih && ih->requires_grad) {
                    ensure_grad(ih);
                    ih->grad[i]        += d_z_raw;
                    ih->grad[oo + i]   += d_r_raw;
                    ih->grad[2*oo + i] += d_n_pre;   /* d_ih_n = d_n_pre */
                }
                if (hh && hh->requires_grad) {
                    ensure_grad(hh);
                    hh->grad[i]        += d_z_raw;
                    hh->grad[oo + i]   += d_r_raw;
                    hh->grad[2*oo + i] += d_hh_n;
                }
                if (prev && prev->requires_grad) {
                    ensure_grad(prev);
                    prev->grad[i] += dh * zv;
                }
            }
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
               d_a = grad @ b^T, d_b = a^T @ grad. beta=1.0 accumulates. */
            int mm = a->shape[0], nn = a->shape[1], kk = r->shape[1];
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
#ifdef __APPLE__
                /* d_a [m,n] = grad [m,k] @ b^T [k,n] */
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            mm, nn, kk, 1.0,
                            r->grad, kk, b->data, kk,
                            1.0, a->grad, nn);
#else
                for (int i = 0; i < mm; i++)
                    for (int j = 0; j < nn; j++) {
                        double s = 0;
                        for (int p = 0; p < kk; p++) s += r->grad[i*kk+p] * b->data[j*kk+p];
                        a->grad[i*nn+j] += s;
                    }
#endif
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
#ifdef __APPLE__
                /* d_b [n,k] = a^T [n,m] @ grad [m,k] */
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            nn, kk, mm, 1.0,
                            a->data, nn, r->grad, kk,
                            1.0, b->grad, kk);
#else
                for (int j = 0; j < nn; j++)
                    for (int p = 0; p < kk; p++) {
                        double s = 0;
                        for (int i = 0; i < mm; i++) s += a->data[i*nn+j] * r->grad[i*kk+p];
                        b->grad[j*kk+p] += s;
                    }
#endif
            }
            break;
        }

        case OP_BMM: {
            /* r = a @ b where a=[B,m,n], b=[n,k], r=[B,m,k]
               d_a[bi] = grad[bi] @ b^T, d_b = sum_bi a[bi]^T @ grad[bi].
               b is shared across batch, so collapse [B,m,*] to [B*m,*] for d_b. */
            int BB = a->shape[0], mm = a->shape[1], nn = a->shape[2], kk = b->shape[1];
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
#ifdef __APPLE__
                /* d_a [B*m, n] = grad [B*m, k] @ b^T [k, n] — one big dgemm */
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            BB * mm, nn, kk, 1.0,
                            r->grad, kk, b->data, kk,
                            1.0, a->grad, nn);
#else
                for (int bi = 0; bi < BB; bi++)
                    for (int i = 0; i < mm; i++)
                        for (int j = 0; j < nn; j++) {
                            double s = 0;
                            for (int p = 0; p < kk; p++)
                                s += r->grad[bi*mm*kk + i*kk+p] * b->data[j*kk+p];
                            a->grad[bi*mm*nn + i*nn+j] += s;
                        }
#endif
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
#ifdef __APPLE__
                /* d_b [n,k] = a^T [n, B*m] @ grad [B*m, k] — single dgemm */
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            nn, kk, BB * mm, 1.0,
                            a->data, nn, r->grad, kk,
                            1.0, b->grad, kk);
#else
                for (int bi = 0; bi < BB; bi++)
                    for (int j = 0; j < nn; j++)
                        for (int p = 0; p < kk; p++) {
                            double s = 0;
                            for (int i = 0; i < mm; i++)
                                s += a->data[bi*mm*nn + i*nn+j] * r->grad[bi*mm*kk + i*kk+p];
                            b->grad[j*kk+p] += s;
                        }
#endif
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
            /* d(Ax)/dA = grad . x^T (rank-1 update), d(Ax)/dx = A^T @ grad */
            MvMeta* meta = (MvMeta*)e->op_meta;
            int m_mv = meta ? meta->m : a->shape[0];
            int n_mv = meta ? meta->n : a->shape[1];
            double* x_vals = meta ? meta->x_vals : b->data;
            ensure_grad(r);
            if (a->requires_grad) {
                ensure_grad(a);
#ifdef __APPLE__
                /* A.grad [m,n] += grad [m] * x^T [n] — rank-1 outer product */
                cblas_dger(CblasRowMajor, m_mv, n_mv, 1.0,
                           r->grad, 1, x_vals, 1,
                           a->grad, n_mv);
#else
                for (int ii = 0; ii < m_mv; ii++)
                    for (int jj = 0; jj < n_mv; jj++)
                        a->grad[ii*n_mv+jj] += r->grad[ii] * x_vals[jj];
#endif
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
#ifdef __APPLE__
                /* x.grad [n] += A^T [n,m] @ grad [m] */
                cblas_dgemv(CblasRowMajor, CblasTrans, m_mv, n_mv, 1.0,
                            a->data, n_mv, r->grad, 1,
                            1.0, b->grad, 1);
#else
                for (int jj = 0; jj < n_mv; jj++) {
                    double s = 0;
                    for (int ii = 0; ii < m_mv; ii++) s += a->data[ii*n_mv+jj] * r->grad[ii];
                    b->grad[jj] += s;
                }
#endif
            }
            break;
        }

        case OP_CONCAT_2D_AXIS1: {
            /* r[m, n+k] = concat(A[m,n], B[m,k]) along axis 1.
               dA[i,j]   += r->grad[i*(n+k) + j] for j<n
               dB[i,j-n] += r->grad[i*(n+k) + j] for j>=n */
            int m_c = a->shape[0];
            int n_c = (int)e->scalar_arg;
            int k_c = b->shape[1];
            ensure_grad(r);
            if (a->requires_grad) {
                ensure_grad(a);
                for (int i = 0; i < m_c; i++)
                    for (int j = 0; j < n_c; j++)
                        a->grad[i*n_c + j] += r->grad[i*(n_c + k_c) + j];
            }
            if (b->requires_grad) {
                ensure_grad(b);
                for (int i = 0; i < m_c; i++)
                    for (int j = 0; j < k_c; j++)
                        b->grad[i*k_c + j] += r->grad[i*(n_c + k_c) + (n_c + j)];
            }
            break;
        }

        case OP_LINEAR_2D: {
            /* Y[B,o] = X[B,i] @ W[o,i]^T + bias[o].
               dW[o,i]   = dY^T [o,B] @ X [B,i]    (== dY^T @ X)
               dX[B,i]   = dY [B,o] @ W [o,i]
               dbias[o]  = sum_b dY[b,o] */
            Linear2dMeta* lm2 = (Linear2dMeta*)e->op_meta;
            int B2 = lm2->B, i2 = lm2->i, o2 = lm2->o;
            double* x_vals_2 = lm2->x_vals;
            ensure_grad(r);
            /* a = W [o,i] */
            if (a->requires_grad) {
                ensure_grad(a);
#ifdef __APPLE__
                /* dW [o,i] = dY^T [o,B] @ X [B,i] */
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            o2, i2, B2, 1.0,
                            r->grad, o2, x_vals_2, i2,
                            1.0, a->grad, i2);
#else
                for (int oo = 0; oo < o2; oo++)
                    for (int jj = 0; jj < i2; jj++) {
                        double s = 0;
                        for (int bb = 0; bb < B2; bb++)
                            s += r->grad[bb*o2+oo] * x_vals_2[bb*i2+jj];
                        a->grad[oo*i2+jj] += s;
                    }
#endif
            }
            /* b = X [B,i] */
            if (b && b->requires_grad) {
                ensure_grad(b);
#ifdef __APPLE__
                /* dX [B,i] = dY [B,o] @ W [o,i] */
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            B2, i2, o2, 1.0,
                            r->grad, o2, a->data, i2,
                            1.0, b->grad, i2);
#else
                for (int bb = 0; bb < B2; bb++)
                    for (int jj = 0; jj < i2; jj++) {
                        double s = 0;
                        for (int oo = 0; oo < o2; oo++)
                            s += r->grad[bb*o2+oo] * a->data[oo*i2+jj];
                        b->grad[bb*i2+jj] += s;
                    }
#endif
            }
            /* bias [o] */
            if (lm2->bias && lm2->bias->requires_grad) {
                ensure_grad(lm2->bias);
                for (int oo = 0; oo < o2; oo++) {
                    double s = 0;
                    for (int bb = 0; bb < B2; bb++) s += r->grad[bb*o2+oo];
                    lm2->bias->grad[oo] += s;
                }
            }
            break;
        }

        case OP_LINEAR: {
            /* y = W @ x + bias.  dW = grad . x^T (rank-1), dx = W^T @ grad, dbias = grad */
            LinearMeta* lm = (LinearMeta*)e->op_meta;
            int m_l = lm->m, n_l = lm->n;
            double* x_vals_l = lm->x_vals;
            ensure_grad(r);
            /* a = W [m,n] */
            if (a->requires_grad) {
                ensure_grad(a);
#ifdef __APPLE__
                cblas_dger(CblasRowMajor, m_l, n_l, 1.0,
                           r->grad, 1, x_vals_l, 1,
                           a->grad, n_l);
#else
                for (int ii = 0; ii < m_l; ii++)
                    for (int jj = 0; jj < n_l; jj++)
                        a->grad[ii*n_l+jj] += r->grad[ii] * x_vals_l[jj];
#endif
            }
            /* b = x [n] */
            if (b && b->requires_grad) {
                ensure_grad(b);
#ifdef __APPLE__
                cblas_dgemv(CblasRowMajor, CblasTrans, m_l, n_l, 1.0,
                            a->data, n_l, r->grad, 1,
                            1.0, b->grad, 1);
#else
                for (int jj = 0; jj < n_l; jj++) {
                    double s = 0;
                    for (int ii = 0; ii < m_l; ii++) s += a->data[ii*n_l+jj] * r->grad[ii];
                    b->grad[jj] += s;
                }
#endif
            }
            /* bias */
            if (lm->bias && lm->bias->requires_grad) {
                ensure_grad(lm->bias);
                for (int ii = 0; ii < m_l; ii++)
                    lm->bias->grad[ii] += r->grad[ii];
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

        case OP_CONV2D_BATCHED: {
            /* r = conv2d_batched(input [B,inC,H,W], kernel [outC,inC,kH,kW]) + bias
               r=[B,outC,oH,oW]. Backward via im2col + cblas_dgemm:
                   dY_unf [M, outC] = permute(r.grad, (B,outC,oH,oW) -> (B*oH*oW, outC))
                   dW   [outC, K] = dY_unf^T [outC, M] @ X_col [M, K]   (one dgemm)
                   dX_col [M, K]  = dY_unf [M, outC]   @ W [outC, K]    (one dgemm)
                   dInput += col2im(dX_col)
                   dBias[oc]  += sum_{b,oh,ow} dY_unf[b*oH*oW+oh*oW+ow, oc]
               K = inC*kH*kW, M = B*oH*oW. */
            Conv2DBatchedMeta* meta = (Conv2DBatchedMeta*)e->op_meta;
            int B = meta->B;
            int inC = meta->inC, outC = meta->outC;
            int HH = meta->H, WW = meta->W, kH = meta->kH, kW = meta->kW;
            int padH = meta->padH, padW = meta->padW;
            int strideH = meta->strH, strideW = meta->strW;
            int oH = meta->oH, oW = meta->oW;
            int K_unf = inC * kH * kW;
            int M_unf = B * oH * oW;
            int out_per_sample = outC * oH * oW;
            ensure_grad(r);

            Tensor* bias_t = (Tensor*)e->inputs;
            int need_dW = b && b->requires_grad;
            int need_dX = a && a->requires_grad;
            int need_dB = bias_t && bias_t->requires_grad;

            /* Permute dY [B, outC, oH, oW] -> dY_unf [B*oH*oW, outC] */
            double* dY_unf = (need_dW || need_dX) ?
                (double*)calloc((size_t)M_unf * outC, sizeof(double)) : NULL;
            if (dY_unf) {
                for (int bb = 0; bb < B; bb++) {
                    const double* dout_b = r->grad + (size_t)bb * out_per_sample;
                    for (int oc = 0; oc < outC; oc++) {
                        for (int oh = 0; oh < oH; oh++) {
                            for (int ow = 0; ow < oW; ow++) {
                                int row = bb * oH * oW + oh * oW + ow;
                                dY_unf[row * outC + oc] = dout_b[oc*oH*oW + oh*oW + ow];
                            }
                        }
                    }
                }
            }

            /* d_kernel — single dgemm: dW[outC,K] = dY_unf^T[outC,M] @ X_col[M,K] */
            if (need_dW) {
                ensure_grad(b);
                double* X_col = (double*)calloc((size_t)M_unf * K_unf, sizeof(double));
                conv2d_im2col(a->data, B, inC, HH, WW, kH, kW, padH, padW,
                              strideH, strideW, oH, oW, X_col);
#ifdef __APPLE__
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            outC, K_unf, M_unf, 1.0,
                            dY_unf, outC,
                            X_col, K_unf,
                            1.0, b->grad, K_unf);
#else
                for (int oc = 0; oc < outC; oc++)
                    for (int kk = 0; kk < K_unf; kk++) {
                        double s = 0;
                        for (int m = 0; m < M_unf; m++)
                            s += dY_unf[m*outC + oc] * X_col[m*K_unf + kk];
                        b->grad[oc*K_unf + kk] += s;
                    }
#endif
                free(X_col);
            }

            /* d_input — dX_col[M,K] = dY_unf[M,outC] @ W[outC,K], then col2im */
            if (need_dX) {
                ensure_grad(a);
                double* dX_col = calloc((size_t)M_unf * K_unf, sizeof(double));
#ifdef __APPLE__
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M_unf, K_unf, outC, 1.0,
                            dY_unf, outC,
                            b->data, K_unf,
                            0.0, dX_col, K_unf);
#else
                for (int m = 0; m < M_unf; m++)
                    for (int kk = 0; kk < K_unf; kk++) {
                        double s = 0;
                        for (int oc = 0; oc < outC; oc++)
                            s += dY_unf[m*outC + oc] * b->data[oc*K_unf + kk];
                        dX_col[m*K_unf + kk] = s;
                    }
#endif
                conv2d_col2im_accumulate(dX_col, B, inC, HH, WW, kH, kW,
                                          padH, padW, strideH, strideW,
                                          oH, oW, a->grad);
                free(dX_col);
            }

            /* d_bias — sum across B and (oH, oW) per output channel */
            if (need_dB) {
                ensure_grad(bias_t);
                for (int oc = 0; oc < outC; oc++) {
                    double s = 0;
                    for (int bb = 0; bb < B; bb++) {
                        const double* dout_b = r->grad + (size_t)bb * out_per_sample;
                        for (int oh = 0; oh < oH; oh++)
                            for (int ow = 0; ow < oW; ow++)
                                s += dout_b[oc*oH*oW + oh*oW + ow];
                    }
                    bias_t->grad[oc] += s;
                }
            }
            if (dY_unf) free(dY_unf);
            break;
        }

        case OP_MAX_POOL2D_BATCHED: {
            /* r = max_pool2d_batched(a=input [B,C,H,W]). max_indices are absolute
               into a->data, so direct scatter works the same as the per-sample case. */
            MaxPool2DBatchedMeta* meta = (MaxPool2DBatchedMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int out_numel = meta->B * meta->C * meta->oH * meta->oW;
                for (int i = 0; i < out_numel; i++)
                    a->grad[meta->max_indices[i]] += r->grad[i];
            }
            break;
        }

        case OP_SCATTER_ADD: {
            /* r[index[i]] += a[i]. Backward: d_a[i] += d_r[index[i]] (gather). */
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                Tensor* index = b;  /* b holds the index tensor */
                int nn = a->numel;
                for (int i = 0; i < nn; i++) {
                    int idx = (int)index->data[i];
                    if (idx >= 0 && idx < r->numel)
                        a->grad[i] += r->grad[idx];
                }
            }
            break;
        }
        case OP_GATHER: {
            /* r[i] = a[index[i]]. Backward: d_a[index[i]] += d_r[i] (scatter-add). */
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                Tensor* index = b;  /* b holds the index tensor */
                int nn = (int)e->scalar_arg;
                for (int i = 0; i < nn; i++) {
                    int idx = (int)index->data[i];
                    if (idx >= 0 && idx < a->numel)
                        a->grad[idx] += r->grad[i];
                }
            }
            break;
        }
        case OP_CUMPROD: {
            /* r[i] = prod(a[0..i]). Backward:
               d_a[i] = sum_{j>=i} d_r[j] * r[j] / a[i]
               When a[i] == 0: use exclusive prefix recomputation. */
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int n = a->numel;
                /* Safe backward: compute d_a[i] by accumulating from the right */
                double suffix_sum = 0.0;
                for (int i = n - 1; i >= 0; i--) {
                    suffix_sum += r->grad[i] * r->data[i];
                    if (fabs(a->data[i]) > 1e-30) {
                        a->grad[i] += suffix_sum / a->data[i];
                    } else {
                        /* a[i] == 0: recompute without a[i] */
                        double partial = 0.0;
                        for (int j = i; j < n; j++) {
                            double prod_excl = 1.0;
                            for (int k = 0; k <= j; k++) {
                                if (k != i) prod_excl *= a->data[k];
                            }
                            partial += r->grad[j] * prod_excl;
                        }
                        a->grad[i] += partial;
                    }
                }
            }
            break;
        }

        case OP_LEAKY_RELU: {
            /* d/dx leaky_relu = 1 if x >= 0, alpha otherwise */
            double alpha = e->scalar_arg;
            if (a) {
                ensure_grad(a);
                for (int j = 0; j < a->numel; j++)
                    a->grad[j] += r->grad[j] * (a->data[j] >= 0 ? 1.0 : alpha);
            }
            break;
        }
        case OP_SILU: {
            /* d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))) */
            if (a) {
                ensure_grad(a);
                for (int j = 0; j < a->numel; j++) {
                    double x = a->data[j];
                    double s = 1.0 / (1.0 + exp(-x));
                    a->grad[j] += r->grad[j] * s * (1.0 + x * (1.0 - s));
                }
            }
            break;
        }

        default: break; /* unimplemented backward */
        }
        /* Accumulate per-op timing */
        if (e->op < OP_COUNT) {
            prof_backward_per_op[e->op] += _wall_ms() - t_op;
            prof_backward_count_per_op[e->op]++;
        }
        }  /* close inner _j loop */
    }      /* close outer _cidx loop */
    free(_chunks_b);
    prof_backward_processed += processed;
    prof_backward_skipped += skipped;
    prof_backward_ms += _wall_ms() - t0;
    prof_backward_ops += processed;

    /* Phase 1.5e diagnostic: when DEBUG_PARAM_GRADS is set, dump per-param
       gradient L2 norm to stderr. Use to identify zero/NaN/wrong-magnitude
       grads after a single backward pass. param_registry is declared
       further down in the file; defer to the inner function below. */
    extern void _dbg_dump_param_grads_if_enabled(void);
    _dbg_dump_param_grads_if_enabled();
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

void tensor_no_grad_begin(void) { no_grad_depth++; }
void tensor_no_grad_end(void)   { if (no_grad_depth > 0) no_grad_depth--; }

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
        TapeEntry* e_h = tape_append(OP_LSTM_GATES, (Tensor*)*out_h, combined, prev_cell, (double)o);
        e_h->op_meta = meta;
        /* Record cell output with OP_LSTM_GATES_CELL — backward propagates d_cell.
           Both entries share the same metadata and accumulate gradients additively
           into combined->grad and prev_cell->grad. */
        TapeEntry* e_c = tape_append(OP_LSTM_GATES_CELL, (Tensor*)*out_c, combined, prev_cell, (double)o);
        e_c->op_meta = meta;
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
   GRU Cell
   combined = [z_raw, r_raw, n_raw] each [o], total [3*o]
   z = sigmoid(z_raw), r = sigmoid(r_raw)
   n = tanh(n_raw) -- note: n_raw should already include r*h contribution
   h' = (1-z)*n + z*prev_hidden
   ================================================================ */

TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh, TensorHandle hprev, int o) {
    /* Standard nn.GRU equation. Takes ih = W_ih @ x + b_ih and
       hh = W_hh @ h + b_hh as separate [3*o] vectors (caller's
       responsibility to compute the two halves).
         z = sigmoid(ih_z + hh_z)
         r = sigmoid(ih_r + hh_r)
         n = tanh(ih_n + r * hh_n)
         h' = (1 - z) * n + z * prev
       Pre-2026-05-09 this kernel took a single combined = ih + hh
       and ignored r (simplified GRU); aligned to the standard
       nn.GRU equation so the example matches what library users
       expect. */
    Tensor* ih = (Tensor*)hih;
    Tensor* hh = (Tensor*)hhh;
    Tensor* prev = (Tensor*)hprev;

    double* zG = malloc(o * sizeof(double));
    double* rG = malloc(o * sizeof(double));
    double* nG = malloc(o * sizeof(double));
    double* out = calloc(o, sizeof(double));

    for (int i = 0; i < o; i++) {
        zG[i] = 1.0 / (1.0 + exp(-(ih->data[i] + hh->data[i])));            /* z */
        rG[i] = 1.0 / (1.0 + exp(-(ih->data[o + i] + hh->data[o + i])));    /* r */
        nG[i] = tanh(ih->data[2*o + i] + rG[i] * hh->data[2*o + i]);        /* n */
        out[i] = (1.0 - zG[i]) * nG[i] + zG[i] * prev->data[i];             /* h' */
    }

    int shape[] = {o};
    int rg = ih->requires_grad || hh->requires_grad || prev->requires_grad;
    Tensor* r = make_tensor(out, shape, 1, rg);
    free(out);

    if (r->requires_grad) {
        /* arg1=ih, arg2=hh, prev kept in op_meta (3rd input doesn't fit
           in TapeEntry's two-arg slot). */
        TapeEntry* e = tape_append(OP_GRU_CELL, r, ih, hh, 0);
        GruCellMeta* meta = arena_alloc(sizeof(GruCellMeta));
        meta->o = o;
        meta->zG = zG; meta->rG = rG; meta->nG = nG;
        meta->prev = prev;
        e->op_meta = meta;
    } else {
        free(zG); free(rG); free(nG);
    }
    return r;
}

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

/* Phase 1.5e diagnostic: dump per-param gradient L2 norms after a backward
   pass. Enabled by setting DEBUG_PARAM_GRADS env var. Defined here (rather
   than inline in tensor_backward) so it can see the static param_registry. */
void _dbg_dump_param_grads_if_enabled(void) {
    if (!getenv("DEBUG_PARAM_GRADS")) return;
    fprintf(stderr, "=== param grads after backward ===\n");
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        double l2 = 0.0;
        int has_nan = 0;
        if (t->grad) {
            for (int j = 0; j < t->numel; j++) {
                double g = t->grad[j];
                if (isnan(g) || isinf(g)) has_nan = 1;
                l2 += g * g;
            }
            l2 = sqrt(l2);
        }
        fprintf(stderr, "  %-40s numel=%-6d l2=%12.6e%s%s\n",
                param_registry[i].name, t->numel, l2,
                t->grad ? "" : " NO_GRAD",
                has_nan ? " NAN_OR_INF!" : "");
    }
}

/* Phase 1.5e diagnostic: dump h0/c0 param value trajectories + first 3
   element values. Set DEBUG_LSTM_TRAJ to print every N epochs. */
static int _dbg_traj_step = 0;
void _dbg_dump_lstm_traj_if_enabled(void) {
    if (!getenv("DEBUG_LSTM_TRAJ")) return;
    int every = 100;
    const char* every_s = getenv("DEBUG_LSTM_TRAJ_EVERY");
    if (every_s) every = atoi(every_s);
    _dbg_traj_step++;
    if (_dbg_traj_step % every != 0 && _dbg_traj_step != 1) return;
    for (int i = 0; i < param_count_val; i++) {
        const char* nm = param_registry[i].name;
        /* Match _h0 or _c0 (LSTM learned init) */
        size_t L = strlen(nm);
        if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 || strcmp(nm + L - 3, "_c0") == 0)) {
            Tensor* t = param_registry[i].tensor;
            double l2 = 0.0, mn = 1e300, mx = -1e300;
            for (int j = 0; j < t->numel; j++) {
                double v = t->data[j];
                l2 += v*v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            l2 = sqrt(l2);
            fprintf(stderr, "[traj epoch %d] %s l2=%.10g min=%.10g max=%.10g | t[0..2]=%.10g, %.10g, %.10g\n",
                    _dbg_traj_step, nm, l2, mn, mx,
                    t->numel >= 1 ? t->data[0] : 0.0,
                    t->numel >= 2 ? t->data[1] : 0.0,
                    t->numel >= 3 ? t->data[2] : 0.0);
        }
    }
}

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

TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
    int shape[] = {n};
    return tensor_reshape(h, shape, 1);
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
                TapeEntry* e = tape_append(OP_STACK, r, NULL, NULL, 0);
                e->inputs = inputs;
                e->input_count = count;
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
        TapeEntry* e = tape_append(OP_STACK, r, NULL, NULL, 0);
        e->inputs = inputs;
        e->input_count = count;
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

/* ================================================================
   Per-dtype creation variants
   --------------------------------------------------------------
   Tape's arena is fp64-only (double* throughout) — there's no fp32
   storage path. So _f64 variants delegate to the existing
   unsuffixed creators (current behavior), and _f32 variants are
   abort stubs.

   The abort stubs exist for link-time symbol completeness only:
   when BACKEND=tape, the dylib needs to export every prototype
   declared in backend.h. The Idris-side RuntimeDType F32 instance
   is intentionally not bound to tape's _f32 symbols (Phase 4), so
   F32-typed code can't reach these from typed Idris at all. If a
   caller bypasses the typed surface and hits these directly, the
   abort gives a clear diagnostic instead of silent fp32->fp64
   demotion that would mask the bug.

   When tape gains an fp32 arena (separate workstream — its own
   TODO row), the stubs become real impls. ================================================================ */

#include <stdio.h>

static TensorHandle tape_f32_unsupported(const char* sym) {
    fprintf(stderr,
        "[tape backend] %s called but tape has no fp32 arena. "
        "Bind your code to F64 on tape, or build with BACKEND=mlx / torch.\n",
        sym);
    abort();
}

TensorHandle tensor_create_scalar_f64(double v, int rg)                                 { return tensor_create_scalar(v, rg); }
TensorHandle tensor_create_f64(double* d, int* s, int r, int rg)                        { return tensor_create(d, s, r, rg); }
TensorHandle tensor_create_1d_f64(int n, double* d, int rg)                             { return tensor_create_1d(n, d, rg); }
TensorHandle tensor_create_2d_f64(int rows, int cols, double* d, int rg)                { return tensor_create_2d(rows, cols, d, rg); }
TensorHandle tensor_create_param_1d_f64(int n, double* d)                               { return tensor_create_param_1d(n, d); }
TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* d)                  { return tensor_create_param_2d(rows, cols, d); }
TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* d)              { return tensor_create_param_3d(d0, d1, d2, d); }
TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* d)      { return tensor_create_param_4d(d0, d1, d2, d3, d); }
TensorHandle tensor_create_state_1d_f64(int n, double* d)                               { return tensor_create_state_1d(n, d); }
TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* d)                  { return tensor_create_state_2d(rows, cols, d); }

TensorHandle tensor_create_scalar_f32(double v, int rg)                                 { (void)v; (void)rg; return tape_f32_unsupported("tensor_create_scalar_f32"); }
TensorHandle tensor_create_f32(double* d, int* s, int r, int rg)                        { (void)d; (void)s; (void)r; (void)rg; return tape_f32_unsupported("tensor_create_f32"); }
TensorHandle tensor_create_1d_f32(int n, double* d, int rg)                             { (void)n; (void)d; (void)rg; return tape_f32_unsupported("tensor_create_1d_f32"); }
TensorHandle tensor_create_2d_f32(int rows, int cols, double* d, int rg)                { (void)rows; (void)cols; (void)d; (void)rg; return tape_f32_unsupported("tensor_create_2d_f32"); }
TensorHandle tensor_create_param_1d_f32(int n, double* d)                               { (void)n; (void)d; return tape_f32_unsupported("tensor_create_param_1d_f32"); }
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* d)                  { (void)rows; (void)cols; (void)d; return tape_f32_unsupported("tensor_create_param_2d_f32"); }
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* d)              { (void)d0; (void)d1; (void)d2; (void)d; return tape_f32_unsupported("tensor_create_param_3d_f32"); }
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* d)      { (void)d0; (void)d1; (void)d2; (void)d3; (void)d; return tape_f32_unsupported("tensor_create_param_4d_f32"); }
TensorHandle tensor_create_state_1d_f32(int n, double* d)                               { (void)n; (void)d; return tape_f32_unsupported("tensor_create_state_1d_f32"); }
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* d)                  { (void)rows; (void)cols; (void)d; return tape_f32_unsupported("tensor_create_state_2d_f32"); }

/* Per-dtype cast primitives. Tape has only an F64 arena, so the source
 * dtype is necessarily F64 and the F64 destination is a no-op alias:
 * values are unchanged, the FFI wrapper machinery retains the handle
 * and Idris gets a fresh wrapper around the same C handle. Gradients
 * flow through the source's tape entry naturally — no new tape op is
 * appended since the operation is observationally identity. The F32
 * destination aborts (no fp32 arena). */
TensorHandle tensor_cast_dtype_f64(TensorHandle src)                                     { return src; }
TensorHandle tensor_cast_dtype_f32(TensorHandle src)                                     { (void)src; return tape_f32_unsupported("tensor_cast_dtype_f32"); }

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
    int type; /* 0=SGD, 1=RMSprop, 2=Adam, 3=AdamW */
    double alpha, eps, weight_decay, momentum;
    double beta1, beta2;
    double* v;  /* second moment (RMSprop/Adam) */
    double* m;  /* first moment (Adam) / momentum buffer (RMSprop) */
    int t;      /* step count */
    int allocated;
    double* param_lr;      /* per-param LR overrides (NULL = use opt->lr for all) */
    int param_lr_count;    /* number of entries in param_lr */
    char prefix[128];      /* param-name prefix filter (empty = manages all) */
} Optimizer;

/* Returns 1 if param[i]'s name starts with opt->prefix (or prefix is empty). */
static int opt_owns_param(Optimizer* opt, int i) {
    if (opt->prefix[0] == '\0') return 1;
    return strncmp(param_registry[i].name, opt->prefix, strlen(opt->prefix)) == 0;
}

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

/* Adam that only updates params whose registry name starts with `prefix`.
 * Empty prefix behaves like optimizer_create_adam (manages all params). */
OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                            double eps, const char* prefix) {
    Optimizer* opt = (Optimizer*)optimizer_create_adam(lr, beta1, beta2, eps);
    if (prefix) {
        strncpy(opt->prefix, prefix, sizeof(opt->prefix) - 1);
        opt->prefix[sizeof(opt->prefix) - 1] = '\0';
    }
    return opt;
}

/* Polyak soft update at the param-registry level.
 * For each param P whose name starts with `online_scope`, find the
 * corresponding param Q whose name is `target_scope ++ suffix(P)` and
 * blend Q's data in-place: Q.data ← (1−tau)·Q.data + tau·P.data.
 *
 * Used by SAC: actor / q1 / q2 network params are registered with
 * distinct scope prefixes; target-Q params are registered with
 * `q1_tgt_` / `q2_tgt_` prefixes. One call per target network per
 * training step moves the target toward the online at rate τ.
 * Returns number of param-pairs blended (for sanity checking). */
int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
    if (!online_scope || !target_scope) return 0;
    size_t on_len = strlen(online_scope);
    size_t tg_len = strlen(target_scope);
    int blended = 0;
    double one_minus_tau = 1.0 - tau;
    for (int i = 0; i < param_count_val; i++) {
        const char* on_name = param_registry[i].name;
        if (strncmp(on_name, online_scope, on_len) != 0) continue;
        /* Build target name: target_scope ++ (on_name + on_len). */
        char tgt_name[256];
        size_t suffix_len = strlen(on_name + on_len);
        if (tg_len + suffix_len + 1 > sizeof(tgt_name)) continue;
        memcpy(tgt_name, target_scope, tg_len);
        memcpy(tgt_name + tg_len, on_name + on_len, suffix_len + 1);
        /* Find target param. */
        for (int j = 0; j < param_count_val; j++) {
            if (strcmp(param_registry[j].name, tgt_name) != 0) continue;
            Tensor* on_t = param_registry[i].tensor;
            Tensor* tg_t = param_registry[j].tensor;
            if (on_t->numel != tg_t->numel) break;
            for (int k = 0; k < on_t->numel; k++) {
                tg_t->data[k] = one_minus_tau * tg_t->data[k] + tau * on_t->data[k];
            }
            blended++;
            break;
        }
    }
    return blended;
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
    free(opt->v); free(opt->m); free(opt->param_lr); free(opt);
}

void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
    Optimizer* opt = (Optimizer*)h;
    /* Ensure param_lr array is large enough */
    if (opt->param_lr == NULL || opt->param_lr_count < param_count_val) {
        int new_count = param_count_val > 64 ? param_count_val : 64;
        double* new_lr = realloc(opt->param_lr, new_count * sizeof(double));
        /* Initialize new entries to -1 (sentinel: use base LR) */
        for (int i = opt->param_lr_count; i < new_count; i++) new_lr[i] = -1.0;
        opt->param_lr = new_lr;
        opt->param_lr_count = new_count;
    }
    /* Find param by name and set its LR */
    for (int i = 0; i < param_count_val; i++) {
        if (strcmp(param_registry[i].name, name) == 0) {
            opt->param_lr[i] = lr;
            return;
        }
    }
}

void optimizer_set_lr(OptimizerHandle h, double lr) {
    Optimizer* opt = (Optimizer*)h;
    opt->lr = lr;
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
        if (!opt_owns_param(opt, i)) continue;
        /* Phase 1.5e DIAGNOSTIC: SKIP_LSTM_INIT skips updating params whose
           names end in _h0 or _c0 (LSTM learned initial state). Equivalent
           to keeping them as zero state tensors. Used to localize whether
           the convergence regression is in the gradient values being
           applied to h0/c0 vs being elsewhere. Remove once diagnosed. */
        if (getenv("SKIP_LSTM_INIT")) {
            const char* nm = param_registry[i].name;
            size_t L = strlen(nm);
            if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 ||
                           strcmp(nm + L - 3, "_c0") == 0)) {
                continue;
            }
        }
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        int base = param_element_offset(i);

        /* Per-param LR: use override if set, otherwise base LR */
        double lr = opt->lr;
        if (opt->param_lr && i < opt->param_lr_count && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];

        for (int j = 0; j < t->numel; j++) {
            double g = t->grad[j];
            int idx = base + j;  /* per-element index into optimizer buffers */

            switch (opt->type) {
            case 0: /* SGD */
                t->data[j] -= lr * g;
                break;

            case 1: { /* RMSprop */
                opt->v[idx] = opt->alpha * opt->v[idx] + (1.0 - opt->alpha) * g * g;
                double delta = lr * g / (sqrt(opt->v[idx]) + opt->eps);
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
                t->data[j] -= lr * mhat / (sqrt(vhat) + opt->eps);
                break;
            }

            case 3: { /* AdamW (decoupled weight decay) */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                t->data[j] -= lr * mhat / (sqrt(vhat) + opt->eps);
                t->data[j] -= lr * opt->weight_decay * t->data[j];
                break;
            }
            }
        }
    }

    /* Phase 1.5e: dump h0/c0 trajectory if enabled */
    {
        extern void _dbg_dump_lstm_traj_if_enabled(void);
        _dbg_dump_lstm_traj_if_enabled();
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
    /* Auto-start timing for next epoch's forward pass + per-op accumulator */
    double t_next = _wall_ms();
    prof_epoch_start = t_next;
    prof_op_t_prev = t_next;
}

/* Internal clip-value helper scoped to params owned by `opt`. */
static void clip_grad_value_opt(Optimizer* opt, double max_val) {
    for (int i = 0; i < param_count_val; i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) {
            if (t->grad[j] > max_val) t->grad[j] = max_val;
            if (t->grad[j] < -max_val) t->grad[j] = -max_val;
        }
    }
}

/* Internal clip-norm helper scoped to params owned by `opt`. */
static double clip_grad_norm_opt(Optimizer* opt, double max_norm) {
    double total = 0;
    for (int i = 0; i < param_count_val; i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) total += t->grad[j] * t->grad[j];
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count_val; i++) {
            if (opt && !opt_owns_param(opt, i)) continue;
            Tensor* t = param_registry[i].tensor;
            if (!t->grad) continue;
            for (int j = 0; j < t->numel; j++) t->grad[j] *= scale;
        }
    }
    return norm;
}

/* Public global clippers retained for direct-FFI callers (backward compat). */
void optimizer_clip_grad_value(double max_val) {
    clip_grad_value_opt(NULL, max_val);
}

double optimizer_clip_grad_norm(double max_norm) {
    return clip_grad_norm_opt(NULL, max_norm);
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
    int tape_chunks = 0;
    for (TypedArenaChunk* c = tape_arena.head; c; c = c->next) tape_chunks++;
    size_t tape_cap_entries = (size_t)tape_chunks * tape_arena.chunk_capacity;
    fprintf(stderr, "  Tape: %d entries (%d chunks × %d cap), %zuKB\n",
            tape_size, tape_chunks, tape_arena.chunk_capacity,
            tape_cap_entries * sizeof(TapeEntry) / 1024);
    fprintf(stderr, "  Params: %d tensors, %d elements, %zuKB grads\n",
            param_count_val, total_param_elems, param_grad_bytes / 1024);
    fprintf(stderr, "  Persistent scalars: %d (~%zuKB leaked)\n",
            persistent_scalar_count, leaked_bytes / 1024);
    fprintf(stderr, "  RSS: peak=%dMB cur=%dMB\n",
            get_rss_mb(), get_current_rss_mb());
    fprintf(stderr, "  Expected: arena %zuKB + tape %zuKB + params %zuKB + leaked %zuKB = %zuKB\n",
            total_cap / 1024,
            tape_cap_entries * sizeof(TapeEntry) / 1024,
            (size_t)total_param_elems * sizeof(double) / 1024,
            leaked_bytes / 1024,
            (total_cap + tape_cap_entries * sizeof(TapeEntry) +
             (size_t)total_param_elems * sizeof(double) + leaked_bytes) / 1024);
}

/* ================================================================
   Profiling
   ================================================================ */

void backend_epoch_begin(void) {
    double t = _wall_ms();
    prof_epoch_start = t;
    prof_op_t_prev = t;
}

void backend_profile_reset(void) {
    prof_forward_ms = prof_backward_ms = prof_optimizer_ms = 0;
    prof_forward_ops = prof_backward_ops = prof_epochs = 0;
    prof_epoch_start = 0;
    prof_op_t_prev = 0;
    prof_backward_processed = prof_backward_skipped = 0;
    memset(prof_backward_per_op, 0, sizeof(prof_backward_per_op));
    memset(prof_backward_count_per_op, 0, sizeof(prof_backward_count_per_op));
    memset(prof_forward_per_op, 0, sizeof(prof_forward_per_op));
    memset(prof_forward_count_per_op, 0, sizeof(prof_forward_count_per_op));
    memset(prof_kernel_per_op, 0, sizeof(prof_kernel_per_op));
    memset(prof_kernel_count_per_op, 0, sizeof(prof_kernel_count_per_op));
    memset(prof_binop_inside_ms, 0, sizeof(prof_binop_inside_ms));
    memset(prof_binop_inside_count, 0, sizeof(prof_binop_inside_count));
    memset(prof_binop_path_count, 0, sizeof(prof_binop_path_count));
    prof_binop_general_ms = 0;
}

static const char* op_name(int op) {
    static const char* names[] = {
        "CONST", "ADD", "SUB", "MUL", "DIV",
        "NEG", "ABS", "EXP", "LOG", "SQRT", "POW",
        "SIGMOID", "TANH",
        "MV", "LINEAR", "DOT", "OUTER",
        "SOFTMAX", "LOG_SOFTMAX",
        "SUM", "MEAN",
        "BCE_LOGITS",
        "LSTM_GATES",
        "ADD_S", "MUL_S", "CLAMP",
        "COS_SIM", "CONV1D_CIRC",
        "LSTM_CELL",
        "STACK", "RESHAPE", "SELECT", "VECMAT", "CAT", "NARROW",
        "LOG_SM_2D",
        "MM", "TRANS_2D", "SM_2D", "MASK_FILL", "LN_2D",
        "BMM", "BMM_3X3", "SM_3D", "TRANS_L2",
        "GELU", "GRU", "EMBED", "BATCH_NORM", "DROPOUT",
        "AVGP1D", "AVGP2D", "CONV1D", "MAXP1D", "CONV2D", "CONV2D_B", "MAXP2D", "MAXP2D_B",
        "CUMPROD", "GATHER", "SCATTER_ADD", "LEAKY_RELU", "SILU",
        "LINEAR_2D", "CONCAT_2D", "SOFTPLUS", "TILE_2D"
    };
    /* Compile-time check: names[] must cover every op tag.
       Add to BOTH this list and the enum when introducing new ops. */
    _Static_assert(sizeof(names)/sizeof(names[0]) == OP_COUNT,
                   "op_name names[] out of sync with OP_COUNT — add new ops here");
    if (op >= 0 && op < OP_COUNT) return names[op];
    return "???";
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
    /* Tape walk stats */
    int total_visited = prof_backward_processed + prof_backward_skipped;
    if (total_visited > 0) {
        fprintf(stderr, "  Backward walk: %d processed, %d skipped (%.0f%% dead)\n",
                prof_backward_processed, prof_backward_skipped,
                100.0 * prof_backward_skipped / total_visited);
    }
    /* Top-5 ops by backward time */
    fprintf(stderr, "  Top backward ops:\n");
    for (int rank = 0; rank < 5; rank++) {
        int best = -1;
        double best_time = 0;
        for (int j = 0; j < OP_COUNT; j++) {
            if (prof_backward_per_op[j] > best_time) {
                /* Skip already printed */
                int already = 0;
                for (int k = 0; k < rank; k++) {
                    /* Find k-th best again to skip it */
                    double kt = 0; int ki = -1;
                    for (int m = 0; m < OP_COUNT; m++) {
                        if (prof_backward_per_op[m] > kt) { kt = prof_backward_per_op[m]; ki = m; }
                    }
                    /* This naive approach doesn't work for rank>0. Use simpler method. */
                    (void)kt; (void)ki;
                }
                (void)already;
                best = j; best_time = prof_backward_per_op[j];
            }
        }
        if (best < 0 || best_time < 0.001) break;
        fprintf(stderr, "    %-12s %.2fms (%d calls)\n",
                op_name(best), best_time, prof_backward_count_per_op[best]);
        prof_backward_per_op[best] = -1; /* mark as printed (will be reset on next profile_reset) */
    }
    /* Top-10 ops by forward time (broader than backward — more ops contribute) */
    fprintf(stderr, "  Top forward ops:\n");
    for (int rank = 0; rank < 10; rank++) {
        int best = -1;
        double best_time = 0;
        for (int j = 0; j < OP_COUNT; j++) {
            if (prof_forward_per_op[j] > best_time) {
                best = j; best_time = prof_forward_per_op[j];
            }
        }
        if (best < 0 || best_time < 0.001) break;
        int n = prof_forward_count_per_op[best];
        double per_call_us = n > 0 ? (best_time * 1000.0 / n) : 0.0;
        fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call)\n",
                op_name(best), best_time, n, per_call_us);
        prof_forward_per_op[best] = -1; /* mark as printed */
    }
    /* Kernel-only timer (elementwise vDSP path only, today). Shows the
       actual kernel time independent of the tape_append attribution. */
    int any_kernel = 0;
    for (int j = 0; j < OP_COUNT; j++) {
        if (prof_kernel_count_per_op[j] > 0) { any_kernel = 1; break; }
    }
    if (any_kernel) {
        fprintf(stderr, "  Direct-kernel timing (subset of ops):\n");
        for (int j = 0; j < OP_COUNT; j++) {
            int n = prof_kernel_count_per_op[j];
            if (n == 0) continue;
            double per_call_us = prof_kernel_per_op[j] * 1000.0 / n;
            fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call) [kernel only]\n",
                    op_name(j), prof_kernel_per_op[j], n, per_call_us);
        }
    }
    fprintf(stderr, "  binop_elementwise paths: fast=%d scalar_bcast=%d general_bcast=%d  general_bcast_total=%.2fms\n",
            prof_binop_path_count[0], prof_binop_path_count[1],
            prof_binop_path_count[2], prof_binop_general_ms);
    int any_inside = 0;
    for (int j = 0; j < OP_COUNT; j++) {
        if (prof_binop_inside_count[j] > 0) { any_inside = 1; break; }
    }
    if (any_inside) {
        fprintf(stderr, "  binop_elementwise inside (entry-to-exit):\n");
        for (int j = 0; j < OP_COUNT; j++) {
            int n = prof_binop_inside_count[j];
            if (n == 0) continue;
            double per_call_us = prof_binop_inside_ms[j] * 1000.0 / n;
            fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call) [in-function]\n",
                    op_name(j), prof_binop_inside_ms[j], n, per_call_us);
        }
    }
}

/* ================================================================
   Debug
   ================================================================ */

const char* backend_name(void) { return "tape"; }

/* Job 3 Phase B — mx::compile is mlx-only; tape backend always reports
   disabled regardless of MLX_COMPILE env var. */
int  tensor_mlx_compile_enabled(void) { return 0; }
int  tensor_mlx_compile_invocations(void) { return 0; }
void tensor_mlx_compile_reset_stats(void) { }

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

/* ================================================================
   Portable FFI helpers (for RefC compatibility)
   ================================================================ */

TensorHandle tensor_backward_return(TensorHandle t) {
    tensor_backward(t);
    return t;
}

TensorHandle param_register_return(const char* name, TensorHandle t) {
    tensor_set_requires_grad(t, 1);
    param_register(name, t);
    return t;
}

int param_zero_all_grads_return(int dummy) {
    (void)dummy;
    param_zero_all_grads();
    return 0;
}

TensorHandle tensor_write_double_return(TensorHandle buf, int off, double val) {
    tensor_write_double(buf, off, val);
    return buf;
}

void* tensor_ptr_array_set_return(void* arr, int idx, TensorHandle t) {
    tensor_ptr_array_set(arr, idx, t);
    return arr;
}

int* tensor_alloc_ints(int n) {
    return (int*)calloc(n, sizeof(int));
}

int* tensor_write_int_return(int* buf, int off, int val) {
    buf[off] = val;
    return buf;
}

int tensor_backward_conditional(TensorHandle t) {
    if (tensor_requires_grad(t))
        tensor_backward(t);
    return param_count();
}

double tensor_backward_return_loss(TensorHandle loss_ptr, double loss_val) {
    if (tensor_requires_grad(loss_ptr))
        tensor_backward(loss_ptr);
    return loss_val;
}

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                         TensorHandle loss_ptr, double loss_val) {
    Tensor* loss = (Tensor*)loss_ptr;
    optimizer_zero_grad(opt);
    if (loss->requires_grad) tensor_backward(loss_ptr);
    /* Scope grad-clipping to this optimizer's owned params, so multi-
     * optimizer setups (SAC actor/q1/q2) each clip their own norm. */
    if (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    return loss_val;
}

int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
    (void)dummy;
    if (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    optimizer_zero_grad(opt);
    return 0;
}


void* idrisml_seq(void* a, void* b) {
    (void)a;
    return b;
}

int backend_memory_report_return(int dummy) {
    (void)dummy;
    backend_memory_report();
    return dummy;
}

int backend_reset_for_eval_return(int dummy) {
    (void)dummy;
    backend_reset_for_eval();
    return dummy;
}

int backend_profile_reset_return(int dummy) {
    (void)dummy;
    backend_profile_reset();
    return dummy;
}

int backend_profile_report_return(int dummy) {
    (void)dummy;
    backend_profile_report();
    return dummy;
}

int dropout_random_seed(int x) {
    return rand() % (x + 1);
}


/* ---- L60 dtype-cascade stream wrappers (no-op stream on tape) ---- */

TensorHandle tensor_create_scalar_f32_streamed(double value, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_scalar_f32(value, requires_grad);
}
TensorHandle tensor_create_scalar_f64_streamed(double value, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_scalar_f64(value, requires_grad);
}
TensorHandle tensor_create_f32_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_f32(data, shape, rank, requires_grad);
}
TensorHandle tensor_create_f64_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_f64(data, shape, rank, requires_grad);
}
TensorHandle tensor_create_1d_f32_streamed(int n, double* data, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_1d_f32(n, data, requires_grad);
}
TensorHandle tensor_create_1d_f64_streamed(int n, double* data, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_1d_f64(n, data, requires_grad);
}
TensorHandle tensor_create_2d_f32_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_2d_f32(rows, cols, data, requires_grad);
}
TensorHandle tensor_create_2d_f64_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag) {
    (void)stream_tag;
    return tensor_create_2d_f64(rows, cols, data, requires_grad);
}
TensorHandle tensor_create_param_1d_f32_streamed(int n, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_1d_f32(n, data);
}
TensorHandle tensor_create_param_1d_f64_streamed(int n, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_1d_f64(n, data);
}
TensorHandle tensor_create_param_2d_f32_streamed(int rows, int cols, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_2d_f32(rows, cols, data);
}
TensorHandle tensor_create_param_2d_f64_streamed(int rows, int cols, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_2d_f64(rows, cols, data);
}
TensorHandle tensor_create_param_3d_f32_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_3d_f32(d0, d1, d2, data);
}
TensorHandle tensor_create_param_3d_f64_streamed(int d0, int d1, int d2, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_3d_f64(d0, d1, d2, data);
}
TensorHandle tensor_create_param_4d_f32_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_4d_f32(d0, d1, d2, d3, data);
}
TensorHandle tensor_create_param_4d_f64_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
}
TensorHandle tensor_create_state_1d_f32_streamed(int n, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_state_1d_f32(n, data);
}
TensorHandle tensor_create_state_1d_f64_streamed(int n, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_state_1d_f64(n, data);
}
TensorHandle tensor_create_state_2d_f32_streamed(int rows, int cols, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_state_2d_f32(rows, cols, data);
}
TensorHandle tensor_create_state_2d_f64_streamed(int rows, int cols, double* data, int stream_tag) {
    (void)stream_tag;
    return tensor_create_state_2d_f64(rows, cols, data);
}
TensorHandle tensor_cast_dtype_f32_streamed(TensorHandle src, int stream_tag) {
    (void)stream_tag;
    return tensor_cast_dtype_f32(src);
}
TensorHandle tensor_cast_dtype_f64_streamed(TensorHandle src, int stream_tag) {
    (void)stream_tag;
    return tensor_cast_dtype_f64(src);
}

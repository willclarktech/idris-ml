/* backend_tape/arena.c — Bump-pointer arena allocator + make_tensor
 * variants + ensure_grad + dtype-aware element load/store.
 *
 * Phase 1.0.1 (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 * Currently #included from backend_tape.c (single-TU build); will be
 * compiled as its own TU once Phase 1.0.4 splits the Makefile rule.
 *
 * Intermediate tensors live in the arena (reset in bulk at
 * optimizer_step). Params use regular malloc.
 */

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

/* F32 storage primitives — mirror the F64 versions but allocate float buffers
   and tag the tensor DT_F32. Used by the F32 stamping of the kernel .inc
   (rung 1 onwards). Caller passes the value as `double` for convenience; the
   constructor narrows once. */
static Tensor* make_scalar_f32(double val, int requires_grad) {
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    float* d = arena_alloc(sizeof(float));
    d[0] = (float)val;
    t->data = d;
    t->shape = NULL;
    t->rank = 0;
    t->numel = 1;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->grad = NULL;
    t->persistent = 0;
    t->dtype_tag = DT_F32;
    return t;
}
static Tensor* make_tensor_arena_f32(float* arena_data, int numel, int* shape, int rank, int requires_grad) {
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    t->data = arena_data;
    t->shape = arena_alloc(rank * sizeof(int));
    memcpy(t->shape, shape, rank * sizeof(int));
    t->rank = rank;
    t->numel = numel;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->grad = NULL;
    t->persistent = 0;
    t->dtype_tag = DT_F32;
    return t;
}

/* SFX(make_scalar)/SFX(make_tensor_arena) aliases the .inc resolves through.
   The F64 alias just forwards to the existing functions (they default to
   DT_F64 via zeroed structs). */
static inline Tensor* make_scalar_f64(double val, int rg) { return make_scalar(val, rg); }
static inline Tensor* make_tensor_arena_f64(double* arena_data, int numel, int* shape, int rank, int rg) {
    return make_tensor_arena(arena_data, numel, shape, rank, rg);
}

/* Grad allocator — grads stay F64 regardless of param dtype (asymmetric
   data=F32 / grad=F64 mirrors mixed-precision practice and keeps the 67-case
   backward switch dtype-agnostic). Optimizer step reads F64 grads and writes
   F32 data, which forces F32 precision on the result. */
static void ensure_grad(Tensor* t) {
    if (!t->grad) {
        t->grad = calloc(t->numel, sizeof(double));
    }
}

/* Dtype-aware element load — returns t->data[i] cast to double, dispatching
   on dtype_tag. Used by tensor_sum / non-routed forward ops that need a
   uniform F64 view + by elementwise backward cases that read input data
   (OP_MUL/DIV/POW/ABS/EXP/LOG/SQRT/SIGMOID/TANH) to handle F32 inputs.
   For F64 (the common case), it's a single double load — same instruction
   count as the prior ((double*)t->data)[i] pattern. */
static inline double tape_load_d(const Tensor* t, int i) {
    return (t->dtype_tag == DT_F32) ? (double)((float*)t->data)[i]
                                    : ((double*)t->data)[i];
}

/* Dtype-aware element store — write `v` into t->data[i], narrowing to
   float when the tensor is F32-tagged. Used by the optimizer step's
   writeback so F32 params stay F32-exact after every update. */
static inline void tape_store_d(Tensor* t, int i, double v) {
    if (t->dtype_tag == DT_F32) ((float*)t->data)[i] = (float)v;
    else                        ((double*)t->data)[i] = v;
}

/* backend_tape/arena.c — Bump-pointer arena allocator + make_tensor
 * variants + ensure_grad. Hot-path single-line load/store inlined via
 * arena.h (static inline).
 *
 * Standalone TU compiled into backend_tape_arena.o.
 *
 * Intermediate tensors live in the arena (reset in bulk at
 * optimizer_step). Params use regular malloc.
 */

#include <stdlib.h>
#include <string.h>
#include "arena.h"

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

void* arena_alloc(size_t bytes) {
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

void arena_reset(void) {
    ArenaChunk* c = arena_head;
    while (c) { c->used = 0; c = c->next; }
    arena_current = arena_head;
}

/* make_scalar/make_tensor: use arena for intermediate tensors */

Tensor* make_scalar(double val, int requires_grad) {
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

Tensor* make_tensor(double* data, int* shape, int rank, int requires_grad) {
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

Tensor* make_tensor_arena(double* arena_data, int numel, int* shape, int rank, int requires_grad) {
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

Tensor* make_scalar_f32(double val, int requires_grad) {
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

Tensor* make_tensor_arena_f32(float* arena_data, int numel, int* shape, int rank, int requires_grad) {
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

/* Arena-allocated zero tensor of arbitrary rank + shape + dtype.
 * Callers (BLAS-backed kernels) use this when an input has a zero-width
 * dimension — cblas_*gemm rejects lda=0, but the mathematical answer is
 * a properly-shaped zero tensor. Skip tape_append on the caller side:
 * a constant-zero result has zero gradient w.r.t. its inputs. */
Tensor* tape_zero_tensor(int* shape, int rank, int dtype_tag, int requires_grad) {
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    Tensor* t = arena_alloc(sizeof(Tensor));
    memset(t, 0, sizeof(Tensor));
    size_t elem_size = (dtype_tag == DT_F32) ? sizeof(float) : sizeof(double);
    t->data = arena_alloc((size_t)numel * elem_size);
    if (numel > 0) memset(t->data, 0, (size_t)numel * elem_size);
    t->shape = arena_alloc((size_t)rank * sizeof(int));
    memcpy(t->shape, shape, (size_t)rank * sizeof(int));
    t->rank = rank;
    t->numel = numel;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->grad = NULL;
    t->persistent = 0;
    t->dtype_tag = dtype_tag;
    return t;
}

/* Grad allocator — grads stay F64 regardless of param dtype (asymmetric
   data=F32 / grad=F64 mirrors mixed-precision practice and keeps the 67-case
   backward switch dtype-agnostic). Optimizer step reads F64 grads and writes
   F32 data, which forces F32 precision on the result. */
void ensure_grad(Tensor* t) {
    if (!t->grad) {
        t->grad = calloc(t->numel, sizeof(double));
    }
}

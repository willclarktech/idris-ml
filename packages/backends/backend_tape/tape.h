/* backend_tape/tape.h — OP_* enum + TapeEntry + per-op meta structs.
 *
 * Public surface of the tape data structure: anything an op file needs to
 * reference (its meta layout, the OP_* tag it appends with).
 *
 * Per-op meta structs currently co-located here. Subsequent commits will move
 * each into its op's source file (e.g. LstmGatesMeta → backend_tape/nn/
 * recurrent/lstm_gates_pair.c).
 */

#ifndef IDRISML_BACKEND_TAPE_TAPE_H
#define IDRISML_BACKEND_TAPE_TAPE_H

#include "tensor.h"

/* Operation tags. OP_COUNT is the sentinel; the op-dispatch table sizes
   itself by this. */
enum {
	OP_CONST = 0,
	OP_ADD,
	OP_SUB,
	OP_MUL,
	OP_DIV,
	OP_NEG,
	OP_ABS,
	OP_EXP,
	OP_LOG,
	OP_SQRT,
	OP_POW,
	OP_SIGMOID,
	OP_TANH,
	OP_MV,
	OP_LINEAR,
	OP_DOT,
	OP_OUTER,
	OP_SOFTMAX,
	OP_LOG_SOFTMAX,
	OP_SUM,
	OP_MEAN,
	OP_BCE_WITH_LOGITS,
	OP_LSTM_GATES,
	OP_ADD_SCALAR,
	OP_MUL_SCALAR,
	OP_CLAMP_MIN,
	OP_COSINE_SIM,
	OP_CONV1D_CIRC,
	OP_LSTM_GATES_CELL,    /* cell output — shares LstmGatesMeta with OP_LSTM_GATES */
	OP_STACK,              /* stack of scalar tensors into 1D */
	OP_RESHAPE,            /* reshape (view) — grad passes through unchanged */
	OP_SELECT,             /* select element from vector — grad goes to parent[index] */
	OP_VECMAT,             /* [n] x [n,m] -> [m] vector-matrix multiply */
	OP_CAT,                /* concatenate two 1D tensors: [a] ++ [b] -> [a+b] */
	OP_NARROW,             /* view into a slice of a 1D tensor */
	OP_LOG_SOFTMAX_2D,     /* row-wise log-softmax on [m,n] */
	OP_MM,                 /* [m,n] x [n,k] -> [m,k] matrix-matrix multiply */
	OP_TRANSPOSE_2D,       /* [m,n] -> [n,m] transpose */
	OP_SOFTMAX_2D,         /* row-wise softmax on [m,n] */
	OP_MASKED_FILL,        /* fill masked positions with a value */
	OP_LAYER_NORM_2D,      /* row-wise layer normalization on [m,n] */
	OP_BMM,                /* batched matrix multiply: [B,m,n] x [n,k] -> [B,m,k] */
	OP_BMM_3X3,            /* batched matmul: [B,m,n] x [B,n,k] -> [B,m,k] */
	OP_SOFTMAX_3D,         /* row-wise softmax on [B,m,n] along last dim */
	OP_TRANSPOSE_LAST2,    /* [B,m,n] -> [B,n,m] */
	OP_GELU,               /* GELU activation (tanh approximation) */
	OP_GRU_CELL,           /* GRU cell: z,r,n gates -> new hidden */
	OP_EMBEDDING,          /* row gather from weight matrix */
	OP_BATCH_NORM,         /* per-channel normalization */
	OP_DROPOUT,            /* inverted dropout with stored mask */
	OP_AVG_POOL1D,         /* [C,L] -> [C,oL] mean pooling */
	OP_AVG_POOL2D,         /* [C,H,W] -> [C,oH,oW] mean pooling */
	OP_CONV1D,             /* [inC,L] * [outC,inC,kL] + [outC] -> [outC,oL] */
	OP_MAX_POOL1D,         /* [C,L] -> [C,oL] with max indices */
	OP_CONV2D,             /* [inC,H,W] * [outC,inC,kH,kW] + [outC] -> [outC,oH,oW] */
	OP_CONV2D_BATCHED,     /* [B,inC,H,W] * [outC,inC,kH,kW] + [outC] -> [B,outC,oH,oW] */
	OP_MAX_POOL2D,         /* [C,H,W] -> [C,oH,oW] with max indices */
	OP_MAX_POOL2D_BATCHED, /* [B,C,H,W] -> [B,C,oH,oW] with max indices */
	OP_CUMPROD,            /* cumulative product along dim 0 */
	OP_GATHER,             /* gather by index: out[i] = input[index[i]] */
	OP_SCATTER_ADD,        /* scatter add: out[index[i]] += src[i] */
	OP_GATHER_ROWS,        /* row-wise gather: out[i] = input[i*n + index[i]] */
	OP_MAX_ROWS,           /* row-wise max: out[i] = max_j input[i*n + j] */
	OP_LEAKY_RELU,         /* max(alpha*x, x) — alpha in scalar_arg */
	OP_SILU,               /* x * sigmoid(x) (Swish activation) */
	OP_LINEAR_2D,          /* [B,o] = [B,i] @ [o,i]^T + [o] (batched fused linear) */
	OP_CONCAT_2D_AXIS1,    /* [m,n] ++ [m,k] -> [m,n+k] along axis 1 */
	OP_SOFTPLUS,           /* log(1 + exp(x)), backward = sigmoid(x) */
	OP_TILE_2D,            /* [m,n] -> [m*rep0, n*rep1]; reps in scalar_arg via 2 int fields */
	OP_CAST_DTYPE,         /* dtype cast — locally linear; identity grad flow back to source */
	OP_RMS_NORM_2D,        /* row-wise RMS normalization on [m,n] (HF LlamaRMSNorm) */
	OP_SWIGLU_2D,          /* silu(gate) * up on [m,n]; gate -> arg1, up -> arg2 */
	OP_COUNT               /* sentinel — must be last */
};

typedef struct {
	int op;
	Tensor* result;    /* non-owning pointer to the result tensor */
	Tensor* arg1;      /* non-owning: first input */
	Tensor* arg2;      /* non-owning: second input (NULL for unary) */
	double scalar_arg; /* for add_scalar, mul_scalar */
	Tensor** inputs;   /* for OP_STACK: array of constituent scalar tensors */
	int input_count;   /* number of inputs for stack */
	void* op_meta;     /* op-specific metadata for backward (arena-allocated) */
} TapeEntry;

/* Op metadata structs for fused backward */
typedef struct {
	int m, n;
	double* x_vals;
} MvMeta;
typedef struct {
	int m, n;
	double* x_vals;
	Tensor* bias;
} LinearMeta;
typedef struct {
	int B, i, o;
	double* x_vals;
	Tensor* bias;
} Linear2dMeta;
typedef struct {
	int n;
	double* out_vals;
} SoftmaxMeta;
/* LstmGatesMeta: layout shared with nn/recurrent/lstm_gates_pair.c.
   Kept here because tape_reset in tape.c calls free() on iG/fG/gG/oG/
   new_cell when finalizing OP_LSTM_GATES tape entries. */
typedef struct {
	int o;
	double* iG;
	double* fG;
	double* gG;
	double* oG;
	double* new_cell;
} LstmGatesMeta;
typedef struct {
	int n, w;
	double key_norm;
	double* row_norms;
	double* dots;
} CosSimMeta;

typedef struct {
	Tensor* gamma; /* scale parameter [n] */
	Tensor* bias;  /* shift parameter [n] */
	double* x_hat; /* normalized values [m*n] */
	double* rstd;  /* reciprocal std devs [m] */
	int m, n;
} LayerNormMeta;

/* RmsNormMeta: row-wise RMSNorm (no centering, no bias). Caches the
   normalized values (x * rstd) and rstd per row for the backward pass.
   See nn/norm/rms_norm_2d.c. */
typedef struct {
	Tensor* weight; /* scale parameter [n] */
	double* x_hat;  /* normalized values x[i,j] * rstd[i] [m*n] */
	double* rstd;   /* reciprocal RMS per row [m] */
	int m, n;
} RmsNormMeta;

/* SwiGluMeta: caches sigmoid(gate) per element so backward avoids
   re-evaluating exp(). Used by nn/activation/swiglu_2d.c. */
typedef struct {
	double* sig_g; /* sigmoid(gate[i,j]) [m*n] */
	int m, n;
} SwiGluMeta;

/* GruCellMeta: layout shared with nn/recurrent/gru_cell.c.
   Kept here because tape_reset in tape.c frees zG/rG/nG. */
typedef struct {
	int o;
	double* zG;
	double* rG;
	double* nG;
	Tensor* prev;
} GruCellMeta;

typedef struct {
	int n, embedDim;
	int* indices; /* [n] integer indices, heap-allocated */
} EmbeddingMeta;

typedef struct {
	Tensor* gamma;
	Tensor* beta;
	double* x_hat; /* normalized values [C * spatial], heap-allocated */
	double* rstd;  /* reciprocal std devs [C], heap-allocated */
	int C, spatial;
} BatchNormMeta;

typedef struct {
	double* mask; /* [numel] binary mask (0 or 1/(1-p)), heap-allocated */
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
	int* max_indices; /* [C * oH * oW] index into flat input per-channel */
} MaxPool2DMeta;

typedef struct {
	int B, C, H, W, kH, kW, strH, strW, oH, oW;
	int* max_indices; /* [B * C * oH * oW] flat-input index per (b, c, oh, ow) */
} MaxPool2DBatchedMeta;

/* TypedArena: fixed-element-size linked-list arena. Backend the tape
 * is built on. Structure exposed here (rather than opaque) so the
 * `tape_size` macro below resolves at every call site without an
 * accessor function. */
typedef struct TypedArenaChunk {
	void* data;
	struct TypedArenaChunk* next;
} TypedArenaChunk;

typedef struct TypedArena {
	TypedArenaChunk* head;
	TypedArenaChunk* tail;
	int size;
	int tail_count;
	int chunk_capacity;
	size_t element_size;
} TypedArena;

/* The tape's per-chunk element capacity. Backward (still in
 * backend_tape.c for now) uses this to walk chunks by index. */
#define TAPE_CHUNK_SIZE (1 << 16)

/* The tape itself — globally shared, defined in tape.c. */
extern TypedArena tape_arena;

/* Convenience for read-heavy call sites in backward / profiling. */
#define tape_size (tape_arena.size)

/* Tape operations (now extern, was static). */

/* Append a forward-op entry; returns the new TapeEntry (or a writable
 * scratch buffer if we're inside withNoGrad and result is non-grad-tracked). */
TapeEntry* tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg);

/* Tear down per-entry heap allocations (op_meta, inputs, grad arrays
 * on non-persistent tensors) and reset tape size to 0. Keeps chunks
 * allocated for reuse. Also calls arena_reset. */
void tape_reset(void);

/* Globals exposed for backward / optimizer / no_grad. */
extern long g_tape_peak;
extern int no_grad_depth;

#endif /* IDRISML_BACKEND_TAPE_TAPE_H */

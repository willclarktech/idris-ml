/* Tape mechanics for the mlx backend's modular tree.
 *
 * mlx's autograd is replay-based: forward ops append an entry to a
 * Wengert tape recording (op_code, result_ptr, arg1_ptr, arg2_ptr,
 * scalar_arg, meta_ptr); backward replays the tape inside mx::vjp to
 * propagate gradients through the captured graph. Per-op .cpp files
 * call tape_append after constructing the forward result.
 *
 * The OP_* enum below assigns each tensor op a stable integer id.
 * Backward replay (defined in the autograd section of backend_mlx.cpp)
 * dispatches on this id to apply the matching local gradient rule.
 *
 * The *ReplayMeta structs are heap-allocated by ops that need extra
 * state beyond the (arg1, arg2, scalar) fields — layer-norm's eps +
 * the gamma/beta tensor indices, etc. The TapeEntry::meta pointer is
 * freed by tape_reset on a per-op-code basis.
 */
#ifndef IDRISML_BACKEND_MLX_TAPE_H
#define IDRISML_BACKEND_MLX_TAPE_H

#include <vector>
#include "tensor.h"

enum {
    OP_CONST = 0,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_NEG, OP_EXP, OP_LOG, OP_SQRT,
    OP_SIGMOID, OP_TANH,
    OP_ADD_SCALAR, OP_MUL_SCALAR, OP_CLAMP_MIN,
    OP_SUM, OP_MEAN,
    OP_MM, OP_BMM, OP_TRANSPOSE_2D,
    OP_SOFTMAX_2D, OP_LOG_SOFTMAX_2D,
    OP_MASKED_FILL, OP_LAYER_NORM_2D,
    OP_RESHAPE, OP_NARROW, OP_CAT,
    OP_POW, OP_ABS,
    OP_STACK, OP_OUTER,
    OP_COSINE_SIM, OP_CONV1D_CIRC,
    OP_MV,
    OP_SELECT,
    OP_BMM_3X3,
    OP_SOFTMAX_3D,
    OP_TRANSPOSE_LAST2,
    OP_GELU,
    OP_GRU_CELL,
    OP_EMBEDDING,
    OP_BATCH_NORM,
    OP_DROPOUT,
    OP_AVG_POOL1D,
    OP_AVG_POOL2D,
    OP_CONV1D,
    OP_MAX_POOL1D,
    OP_CONV2D,
    OP_CONV2D_BATCHED,
    OP_MAX_POOL2D,
    OP_MAX_POOL2D_BATCHED,
    OP_CUMPROD,
    OP_GATHER,        /* gather along axis 0 by integer indices */
    OP_SCATTER_ADD,   /* scatter-add along axis 0 by integer indices */
    OP_LEAKY_RELU,
    OP_SILU,
    OP_SUM_DIM,       /* sum along a single axis with optional keepdim */
    OP_CAT_MULTI,     /* n-ary concatenate along given axis */
    OP_LINEAR_2D,     /* Y = X @ W^T + bias, shapes [B,o]=[B,i]@[o,i]^T+[o] */
    OP_CONCAT_2D_AXIS1, /* [m,n] ++ [m,k] -> [m,n+k] along axis 1 */
    OP_SOFTPLUS,      /* log(1 + exp(x)), backward = sigmoid(x) */
    OP_TILE_2D,       /* [m,n] -> [m*rep0, n*rep1]; meta stores (rep0, rep1) */
    OP_CAST_DTYPE,    /* mx::astype to target dtype; scalar_arg encodes target:
                         0.0 = mx::float32, 1.0 = mx::float64, 2.0 = mx::bfloat16. */
    OP_RMS_NORM_2D,   /* row-wise RMS normalization (HF LlamaRMSNorm); meta carries weight pool idx + eps */
    OP_SWIGLU_2D,     /* silu(gate) * up; gate -> arg1, up -> arg2 */
    OP_EMBEDDING_2D,  /* embedding returning [n, embedDim] (no flatten) */
    OP_COUNT          /* sentinel — must be last; sizes the replay dispatch table */
};

/* Lightweight metadata for ops that need extra info during replay.
   No gradient arrays — mx::grad handles backward automatically. */
struct LayerNormReplayMeta {
    int gamma_pool_idx;
    int bias_pool_idx;
    double eps;
};

struct RmsNormReplayMeta {
    int weight_pool_idx;
    double eps;
};

struct LinearReplayMeta {
    int bias_pool_idx;
};

struct BatchNormReplayMeta {
    int gamma_pool_idx;
    int beta_pool_idx;
    int C, spatial;
    double eps;
};

struct Conv1DReplayMeta {
    int pad, stride, inC, L;
    int bias_pool_idx;
};

struct MaxPool1DReplayMeta {
    int C, L, kL, stride, oL;
};

struct Conv2DReplayMeta {
    int padH, padW, strH, strW;
    int inC, H, W;
    int bias_pool_idx;  /* -1 if no bias */
};

struct Conv2DBatchedReplayMeta {
    int padH, padW, strH, strW;
    int B, inC, H, W;
    int bias_pool_idx;  /* -1 if no bias */
};

struct MaxPool2DReplayMeta {
    int C, H, W, kH, kW, strH, strW, oH, oW;
};

struct AvgPool2DReplayMeta {
    int C, H, W, kH, kW, strH, strW, oH, oW;
};

struct MaxPool2DBatchedReplayMeta {
    int B, C, H, W, kH, kW, strH, strW, oH, oW;
};

struct SumDimReplayMeta {
    int dim;       /* normalized to non-negative at forward */
    int keepdim;   /* 0 or 1 */
};

struct GruCellReplayMeta {
    int o;
    int prev_pool_idx;  /* prev hidden state — 3rd input, doesn't fit in arg1/arg2 */
};

struct TapeEntry {
    int op;
    Tensor* result;
    Tensor* arg1;
    Tensor* arg2;
    double scalar_arg;
    void* meta;
};

/* Wengert list — one entry per grad-requiring forward op. Tape is
   defined in backend_mlx.cpp; per-op .cpp files reach it only through
   tape_append (and never directly). */
extern std::vector<TapeEntry> tape;

/* no_grad_depth_mlx — incremented at tensor_no_grad_begin, decremented at
   tensor_no_grad_end. The gating is enforced inside tape_append, so
   per-op TUs only need to call tape_append and never touch the depth
   directly. Definition lives in backend_mlx/training/autograd.cpp
   (co-located with the begin/end functions that mutate it). The `_mlx`
   suffix dodges a tri-link collision with tape's same-named non-static
   global. */
extern int no_grad_depth_mlx;

/* Diagnostic: count of FFI ops appended this epoch. tape_append fires
   once per grad-requiring op in the forward pass. Counts grad-tracked
   ops only; pure-no-grad ops are not in the tape. */
extern long prof_tape_appends_mlx;

/* Append a forward op to the tape; called by every per-op .cpp after
   constructing the result Tensor. Retains result + args (refcount++)
   so the tape holds them until tape_reset. Returns the tape index
   assigned to `result`, or -1 if no_grad_depth > 0 (the caller's
   result is then marked requires_grad=false). */
int tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg);

/* Reset the tape — releases retains on every arg + result and clears the
   per-op meta. Called at optimizer_step end (after eval+param update) and
   at backend_reset_for_eval. Definition lives in backend_mlx.cpp. */
void tape_reset(void);

#endif /* IDRISML_BACKEND_MLX_TAPE_H */

/* backend_tape.c — Tape-based autograd backend implementing backend.h.
 *
 * Each tensor is a scalar double with an index into a global tape.
 * Multi-dimensional tensors are flat double arrays with shape metadata.
 * Forward ops append to the tape; backward walks it in reverse.
 *
 * Design: arena-allocated tape + Accelerate BLAS for linalg.
 */

#include "backend.h"
#include "shared_utils.h"  /* bf16/f16 bit-conv helpers (Phase 4 round-trip) */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <sys/resource.h>
#include <mach/mach.h>
#endif



/* ================================================================
   Tape backend modular includes (Phase 1.0.4: per-TU compile).
   Implementations live in backend_tape subdirs — compiled as separate
   TUs and linked into libidrisml. Headers below give backend_tape.c
   the type defs + extern decls it needs to reference them.
   ================================================================ */
#include "backend_tape/tensor.h"
#include "backend_tape/arena.h"
#include "backend_tape/tape.h"
#include "backend_tape/broadcast.h"
#include "backend_tape/training/autograd/op_dispatch.h"
#include "backend_tape/core/elementwise/_helpers.h"

/* Forward decl: _wall_ms lives later in the profiling section but is
 * called from the elementwise kernel includes (line 311 area). Defined
 * non-static so tape.c (and any future profiling-using TU) can extern
 * it via tape.h. */
double _wall_ms(void);


/* ================================================================
   Lifecycle — Phase 1a.1: extracted to backend_tape/core/lifecycle/
   {create_scalar,create,clone,free,item}.c
   ================================================================ */

/* ================================================================
   Accessors
   ================================================================ */


/* tensor_numel/dim/size, tensor_to_doubles/floats/int64, tensor_dtype_name:
   moved to backend_tape/training/host_io.c (Phase 1e.4). */

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

/* Broadcast helpers — declared in backend_tape/broadcast.h; defined
 * here (still in the monolith) but non-static so per-op files can
 * call them. MAX_BCAST_RANK comes via the header. */

/* True if `a`'s shape exactly matches `r`'s shape (no broadcast). */
int shapes_equal(Tensor* a, Tensor* r) {
    if (a->numel != r->numel || a->rank != r->rank) return 0;
    for (int k = 0; k < a->rank; k++) {
        if (a->shape[k] != r->shape[k]) return 0;
    }
    return 1;
}

/* Compute broadcast output shape from a and b (right-aligned, numpy rules).
   Returns 1 on success, 0 on incompatible shapes. */
int compute_bcast_shape(Tensor* a, Tensor* b,
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
void compute_bcast_strides(Tensor* a, int r_rank, int* r_shape,
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

/* Element-wise binary + unary kernel bodies live in backend_tape_kernels.inc
   so the same source compiles for F64 and (later, step 6) F32 storage.
   Today: single include with SCALAR=double / SFX(name)=name##_f64.
   The kernels (binop_elementwise_inner_f64 / unop_elementwise_f64) are
   the implementation; the wrappers below are the public entry points
   step 7 will tag-dispatch from. */
/* F64 stamping. */
#define SCALAR    double
#define SFX(name) name##_f64
#define VDSP_VADD vDSP_vaddD
#define VDSP_VSUB vDSP_vsubD
#define VDSP_VMUL vDSP_vmulD
#define VDSP_VDIV vDSP_vdivD
#define VDSP_VNEG vDSP_vnegD
#define VV_EXP    vvexp
#define VV_LOG    vvlog
#define VV_SQRT   vvsqrt
#define VV_TANH   vvtanh
#define VV_FABS   vvfabs
#include "backend_tape_kernels.inc"
#undef SCALAR
#undef SFX
#undef VDSP_VADD
#undef VDSP_VSUB
#undef VDSP_VMUL
#undef VDSP_VDIV
#undef VDSP_VNEG
#undef VV_EXP
#undef VV_LOG
#undef VV_SQRT
#undef VV_TANH
#undef VV_FABS

/* F32 stamping — uses cblas_s* / vDSP_* (no `D` suffix) / vv*f forms. */
#define SCALAR    float
#define SFX(name) name##_f32
#define VDSP_VADD vDSP_vadd
#define VDSP_VSUB vDSP_vsub
#define VDSP_VMUL vDSP_vmul
#define VDSP_VDIV vDSP_vdiv
#define VDSP_VNEG vDSP_vneg
#define VV_EXP    vvexpf
#define VV_LOG    vvlogf
#define VV_SQRT   vvsqrtf
#define VV_TANH   vvtanhf
#define VV_FABS   vvfabsf
#include "backend_tape_kernels.inc"
#undef SCALAR
#undef SFX
#undef VDSP_VADD
#undef VDSP_VSUB
#undef VDSP_VMUL
#undef VDSP_VDIV
#undef VDSP_VNEG
#undef VV_EXP
#undef VV_LOG
#undef VV_SQRT
#undef VV_TANH
#undef VV_FABS

TensorHandle binop_elementwise(TensorHandle ha, TensorHandle hb, int op_tag,
                               double (*scalar_fn)(double, double)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner_f64(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}

TensorHandle binop_elementwise_f32_disp(TensorHandle ha, TensorHandle hb, int op_tag,
                                        float (*scalar_fn)(float, float)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner_f32(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}

void tape_abort_mixed_dtype(const char* op) {
    fprintf(stderr,
        "[tape backend] %s: mixed-dtype inputs forbidden — both operands must "
        "share a dtype_tag (cast first via tcast / tensor_cast_dtype_streamed).\n", op);
    abort();
}

static double fn_add(double a, double b) { return a + b; }
static double fn_sub(double a, double b) { return a - b; }
static double fn_mul(double a, double b) { return a * b; }
static double fn_div(double a, double b) { return a / b; }
static double fn_pow(double a, double b) { return pow(a, b); }

/* F32 scalar function counterparts — the F32 stamping of binop_elementwise_inner
   expects float (*)(float, float). */
static float fn_add_f32(float a, float b) { return a + b; }
static float fn_sub_f32(float a, float b) { return a - b; }
static float fn_mul_f32(float a, float b) { return a * b; }
static float fn_div_f32(float a, float b) { return a / b; }
static float fn_pow_f32(float a, float b) { return powf(a, b); }

/* Forward dispatch: both-F32 → F32 stamping; both-F64 → F64 stamping;
   mixed → abort. The F64 path is the default fallthrough so any inference
   dtype that ever leaks (it shouldn't, given the Idris-side Compatible
   gate) doesn't silently take an unintended branch. */
#define TAPE_BINOP_DISPATCH(name, op_tag, fn64, fn32) \
TensorHandle name(TensorHandle a, TensorHandle b) { \
    Tensor* ta = (Tensor*)a; Tensor* tb = (Tensor*)b; \
    if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) { \
        if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype(#name); \
        return binop_elementwise_f32_disp(a, b, op_tag, fn32); \
    } \
    return binop_elementwise(a, b, op_tag, fn64); \
}
/* tensor_add, tensor_sub, tensor_mul, tensor_div, tensor_pow: moved to backend_tape/core/elementwise/ (Phase 1a.2-7). */
#undef TAPE_BINOP_DISPATCH

/* Unary ops: support both scalar (rank 0) and multi-element tensors */
static double fn_neg(double x) { return -x; }
static double fn_abs(double x) { return fabs(x); }
static double fn_exp_d(double x) { return exp(x); }
static double fn_log_d(double x) { return log(x); }
static double fn_sqrt_d(double x) { return sqrt(x); }
static double fn_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double fn_tanh_d(double x) { return tanh(x); }
/* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static double fn_gelu_d(double x) {
    double c = 0.7978845608028654;  /* sqrt(2/pi) */
    double inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}
/* F32 unary scalar functions — paired with unop_elementwise_f32. */
static float fn_neg_f32(float x) { return -x; }
static float fn_abs_f32(float x) { return fabsf(x); }
static float fn_exp_f32(float x) { return expf(x); }
static float fn_log_f32(float x) { return logf(x); }
static float fn_sqrt_f32(float x) { return sqrtf(x); }
static float fn_sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }
static float fn_tanh_f32(float x) { return tanhf(x); }
static float fn_gelu_f32(float x) {
    float c = 0.7978845608028654f;
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

TensorHandle unop_elementwise(TensorHandle ha, int op, double (*fn)(double)) {
    /* Body lives in backend_tape_kernels.inc as unop_elementwise_f64. */
    return unop_elementwise_f64(ha, op, fn);
}

/* Symmetric F32 dispatch wrapper (mirror of binop_elementwise_f32_disp).
 * unop_elementwise_f32 itself is static (in the .inc stamping); this
 * non-static wrapper exposes it to per-op files in core/elementwise/. */
TensorHandle unop_elementwise_f32_disp(TensorHandle ha, int op_tag, float (*fn)(float)) {
    return unop_elementwise_f32(ha, op_tag, fn);
}

/* Forward dispatch: tag-aware unop wrappers — F32 input routes to the F32
   stamping, else falls through to F64. The F32 stamping picks up the
   matching fn_*_f32 helper. */
#define TAPE_UNOP_DISPATCH(name, op_tag, fn64, fn32) \
TensorHandle name(TensorHandle ha) { \
    Tensor* a = (Tensor*)ha; \
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32(ha, op_tag, fn32); \
    return unop_elementwise(ha, op_tag, fn64); \
}
/* tensor_neg/abs/exp/log/sqrt: moved to backend_tape/core/elementwise/ (Phase 1a.6). */
/* tensor_sigmoid, tensor_tanh: moved to backend_tape/core/elementwise/ (Phase 1a.8). */
#undef TAPE_UNOP_DISPATCH
/* tensor_gelu: moved to backend_tape/nn/activation/gelu.c (Phase 1c.2). */

/* LeakyReLU: max(alpha*x, x). Uses scalar_arg to store alpha. F32 forward
   uses real F32 arena storage; backward (OP_LEAKY_RELU) reads a->data via
   tape_load_d so both dtypes share the same case body. */
/* tensor_leaky_relu_f32: moved to backend_tape/nn/activation/leaky_relu.c (Phase 1c.2). */

/* tensor_leaky_relu, tensor_silu: moved to backend_tape/nn/activation/ (Phase 1c.2). */

/* tensor_softplus: moved to backend_tape/core/elementwise/softplus.c (Phase 1a.8). */

/* tensor_add_scalar / tensor_mul_scalar / tensor_clamp_min:
 * moved to backend_tape/core/scalar/ (Phase 1a.9). */

/* ================================================================
   Reduction
   ================================================================ */

/* Reductions — dtype-aware scalar output. tape_load_d covers the data read;
   the dtype-matched make_scalar / make_scalar_f32 sets the result tag.
   Backward (OP_SUM, OP_MEAN) only writes input grads (always F64) and
   doesn't read input data, so the existing cases work for both dtypes. */
/* tensor_sum, tensor_sum_dim, tensor_mean, tensor_min, tensor_max:
 * moved to backend_tape/linear/reduction/ (Phase 1b.3). */

/* ================================================================
   Linear algebra
   ================================================================ */

/* F32 stamping of tensor_mv. F64 grads are kept (asymmetric), so the
   backward case still reads a double* x_vals cache — we convert from
   the F32 vec on store. Output uses make_tensor_arena_f32 + tags F32. */
/* tensor_mv: moved to backend_tape/linear/linalg/mv.c (Phase 1b.4.b). */

/* Fused batched linear: Y[B,o] = X[B,i] @ W[o,i]^T + bias[o].
   Single allocation, single tape entry. W: [o, i], X: [B, i], bias: [o] (or NULL). */
/* tensor_linear_2d: moved to backend_tape/linear/linalg/linear_2d.c (Phase 1b.5). */

/* Concatenate along axis 1: A[m,n] ++ B[m,k] -> [m, n+k].
   Single tape entry; backward scatters dY back to dA / dB by column split. */
/* tensor_concat_2d_axis1: moved to backend_tape/linear/concat/concat_2d_axis1.c (Phase 1b.2.c). */

/* F32 stamping of tensor_linear. F32 mat/vec/bias → F32 output via
   cblas_sgemv + vDSP_vadd. Meta caches x_vals as double* (converted on
   store) so the existing backward case can read it uniformly. */
/* tensor_linear: moved to backend_tape/linear/linalg/linear.c (Phase 1b.5). */

/* tensor_dot: moved to backend_tape/linear/linalg/dot.c (Phase 1b.4). */

/* tensor_matmul: moved to backend_tape/linear/linalg/matmul.c (Phase 1b.5; covers OP_VECMAT). */

/* tensor_outer: moved to backend_tape/linear/linalg/outer.c (Phase 1b.4). */

/* tensor_mm: moved to backend_tape/linear/linalg/mm.c (Phase 1b.5). */

/* Batched matrix-matrix multiply: [B,m,n] x [n,k] -> [B,m,k]
   Weight matrix b is shared across all batch elements. */
/* tensor_bmm, tensor_bmm_3x3: moved to backend_tape/linear/linalg/ (Phase 1b.6). */

/* Softmax over the last dim of a [B,m,n] tensor — F32 + F64 unified path. */
/* tensor_softmax_3d: moved to backend_tape/nn/softmax/softmax_3d.c (Phase 1c.1). */

/* tensor_transpose_last2: moved to backend_tape/linear/linalg/transpose_last2.c (Phase 1b.6). */

/* tensor_reshape_4d, tensor_reshape_3d: moved to backend_tape/linear/shape/ (Phase 1b.9). */

/* tensor_expand_mask: moved to backend_tape/nn/mask/expand_mask.c (Phase 1c.3). */

/* tensor_tile_2d + Tile2dMeta: moved to backend_tape/linear/linalg/tile_2d.c (Phase 1b.6). */

/* Stack B tensors of shape [m, n] into [B, m, n].
   All tensors must have the same shape. No gradient tracking (data tensors). */
TensorHandle tensor_batch(TensorHandle* handles, int count) {
    Tensor* first = (Tensor*)handles[0];
    int elem_size = first->numel;
    int total = count * elem_size;
    int rank = first->rank + 1;
    int is_f32 = (first->dtype_tag == DT_F32);
    for (int i = 1; i < count; i++)
        if (((Tensor*)handles[i])->dtype_tag != first->dtype_tag)
            tape_abort_mixed_dtype("tensor_batch");
    int* shape = malloc(rank * sizeof(int));
    shape[0] = count;
    for (int i = 0; i < first->rank; i++) shape[i+1] = first->shape[i];
    Tensor* r;
    if (is_f32) {
        float* data = arena_alloc(total * sizeof(float));
        for (int i = 0; i < count; i++)
            memcpy(data + i * elem_size, ((Tensor*)handles[i])->data, elem_size * sizeof(float));
        r = make_tensor_arena_f32(data, total, shape, rank, 0);
    } else {
        double* data = malloc(total * sizeof(double));
        for (int i = 0; i < count; i++)
            memcpy(data + i * elem_size, ((Tensor*)handles[i])->data, elem_size * sizeof(double));
        r = make_tensor(data, shape, rank, 0);
        free(data);
    }
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
    size_t es = tape_elem_size(t->dtype_tag);
    TensorHandle* handles = malloc(B * sizeof(TensorHandle));
    for (int i = 0; i < B; i++) {
        Tensor* r = arena_alloc(sizeof(Tensor));
        memset(r, 0, sizeof(Tensor));
        r->data = (char*)t->data + (size_t)(i * elem_size) * es;  /* byte-correct view */
        r->shape = arena_alloc(inner_rank * sizeof(int));
        for (int j = 0; j < inner_rank; j++) r->shape[j] = t->shape[j+1];
        r->rank = inner_rank;
        r->numel = elem_size;
        r->requires_grad = t->requires_grad;
        r->tape_idx = -1;
        r->persistent = 0;
        r->dtype_tag = t->dtype_tag;
        handles[i] = (TensorHandle)r;
    }
    return handles;
}

/* tensor_transpose_2d: moved to backend_tape/linear/linalg/transpose_2d.c (Phase 1b.6). */

/* Row-wise softmax on 2D tensor: [m,n] -> [m,n], each row sums to 1 */
/* tensor_softmax_2d, tensor_log_softmax_2d: moved to backend_tape/nn/softmax/ (Phase 1c.1). */

/* Element-wise multiply for multi-element tensors (same shape) */
TensorHandle tensor_mul_elementwise(TensorHandle ha, TensorHandle hb) {
    return tensor_mul(ha, hb);  /* tensor_mul already handles multi-element via binop_elementwise */
}

/* Sum all elements of a tensor (not just scalar) */
TensorHandle tensor_sum_all(TensorHandle h) {
    return tensor_sum(h);  /* tensor_sum already sums all elements */
}

/* Masked fill: replace positions where mask[i]=1 with value */
/* tensor_masked_fill: moved to backend_tape/nn/mask/masked_fill.c (Phase 1c.3). */


/* Row-wise layer normalization on 2D tensor: y[i,j] = gamma[j] * x_hat[i,j] + beta[j]
   where x_hat[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) */
/* tensor_layer_norm_2d: moved to backend_tape/nn/norm/layer_norm_2d.c (Phase 1c.4). */

/* ================================================================
   Activation / normalization
   ================================================================ */

/* F32 stamping of tensor_softmax. Stable formulation (subtract max). */
/* tensor_softmax, tensor_log_softmax (+ _f32 helpers): moved to
 * backend_tape/nn/softmax/ (Phase 1c.1). */

/* ================================================================
   Loss functions
   ================================================================ */

/* Loss kernels — dtype-aware reads via tape_load_d so F32 inputs are
   honoured; output is an F32 scalar when inputs are F32, else F64. Mixed
   dtype between input and target is rejected. */
/* tensor_bce_with_logits: moved to backend_tape/nn/loss/bce_with_logits.c (Phase 1c.6). */

/* tensor_cross_entropy: moved to backend_tape/nn/loss/cross_entropy.c (Phase 1e.8.a). */
/* tensor_mse_loss:      moved to backend_tape/nn/loss/mse_loss.c      (Phase 1e.8.a). */

/* ================================================================
   NTM-specific compositions
   ================================================================ */

/* tensor_cosine_similarity: moved to backend_tape/nn/attention/cosine_similarity.c (Phase 1c.5). */

/* tensor_conv1d_circular: moved to backend_tape/conv/conv1d_circular.c (Phase 1d.1.c). */

/* ================================================================
   Cross-Attention: Q @ K^T * scale [+ mask] -> softmax -> @ V
   Q [B,seqQ,d], K [B,seqK,d], V [B,seqK,d] -> [B,seqQ,d]
   ================================================================ */

/* tensor_cross_attention: moved to backend_tape/nn/attention/cross_attention.c (Phase 1c.5). */

/* ================================================================
   Embedding: row gather from weight matrix
   weight [vocabSize, embedDim], indices [n] -> output [n * embedDim]
   ================================================================ */

/* tensor_embedding: moved to backend_tape/nn/attention/embedding.c (Phase 1c.5). */

/* ================================================================
   Batch Normalization: per-channel, across spatial dims
   Input treated as [C, spatial]. Normalizes each channel independently.
   ================================================================ */

/* tensor_batch_norm: moved to backend_tape/nn/norm/batch_norm.c (Phase 1c.4). */

/* ================================================================
   Group Normalization: normalize within channel groups
   ================================================================ */

/* tensor_group_norm: moved to backend_tape/nn/norm/group_norm.c (Phase 1e.8.a). */

/* ================================================================
   Dropout: inverted dropout with mask
   ================================================================ */

/* tensor_dropout: moved to backend_tape/nn/norm/dropout.c (Phase 1c.4). */

/* ================================================================
   Gather / Scatter
   ================================================================ */

/* tensor_gather: moved to backend_tape/linear/index/gather.c (Phase 1b.7). */

/* tensor_scatter_add: moved to backend_tape/linear/index/scatter_add.c (Phase 1b.7.b). */

/* ================================================================
   Sort / Scan
   ================================================================ */

/* tensor_argsort, tensor_cumprod (+ argsort comparators):
 * moved to backend_tape/linear/sort/ (Phase 1b.8). */

/* ================================================================
   Average Pooling
   ================================================================ */

/* tensor_avg_pool1d: moved to backend_tape/conv/avg_pool1d.c (Phase 1d.1.a). */

/* tensor_avg_pool2d: moved to backend_tape/conv/avg_pool2d.c (Phase 1d.2.a). */

/* tensor_conv1d: moved to backend_tape/conv/conv1d.c (Phase 1d.1.d). */

/* tensor_max_pool1d: moved to backend_tape/conv/max_pool1d.c (Phase 1d.1.b). */

/* tensor_create_param_3d: moved to backend_tape/training/param_create.c (Phase 1e.3). */

/* tensor_conv_transpose1d / conv_transpose2d: moved to backend_tape/conv/conv_transpose.c (Phase 1e.8). */
/* tensor_conv1d_grouped / conv2d_grouped: moved to backend_tape/conv/conv_grouped.c (Phase 1e.8). */

/* tensor_conv2d: moved to backend_tape/conv/conv2d.c (Phase 1d.2.d). */

/* tensor_conv2d_batched + helpers (conv2d_im2col, conv2d_col2im_accumulate):
   moved to backend_tape/conv/conv2d_batched.c (Phase 1d.2.e). */


/* ================================================================
   MaxPool2D: input [C, H, W] -> [C, oH, oW]
   ================================================================ */

/* tensor_max_pool2d: moved to backend_tape/conv/max_pool2d.c (Phase 1d.2.b). */

/* tensor_max_pool2d_batched: moved to backend_tape/conv/max_pool2d_batched.c (Phase 1d.2.c). */

/* ================================================================
   Shape manipulation
   ================================================================ */

/* tensor_reshape: moved to backend_tape/linear/shape/reshape.c (Phase 1b.1.b). */

/* tensor_unsqueeze, tensor_squeeze: moved to backend_tape/linear/shape/ (Phase 1b.1.d). */

/* tensor_select: moved to backend_tape/linear/shape/select.c (Phase 1b.1). */

/* tensor_stack, tensor_cat: moved to backend_tape/linear/concat/ (Phase 1b.2.a). */

/* tensor_cat2: moved to backend_tape/linear/concat/cat2.c (Phase 1b.2.b). */

/* tensor_narrow: moved to backend_tape/linear/shape/narrow.c (Phase 1b.1.c). */

/* ================================================================
   Autograd — backward pass
   ================================================================ */

/* _wall_ms + all prof_* globals + backend_epoch_begin + backend_profile_reset
   + backend_profile_report (and the op_name[] human-readable table):
   moved to backend_tape/training/profiling.c (Phase 1e.7). */



/* tensor_backward: moved to backend_tape/training/autograd/backward.c (Phase 1e.2). */

/* autograd helpers (tensor_grad, _zero_grad, _requires_grad,
   _set_requires_grad, _detach, _with_grad, _no_grad_*, _epoch_*) and
   the CPU-only device shims moved to backend_tape/training/autograd/
   helpers.c (Phase 1e.1). */

/* ================================================================
   LSTM
   ================================================================ */

/* tensor_lstm_cell:           moved to backend_tape/nn/recurrent/lstm_cell.c       (Phase 1c.7.a). */
/* tensor_lstm_gates / pair:   moved to backend_tape/nn/recurrent/lstm_gates_pair.c (Phase 1c.7.d). */

/* tensor_pair_first/second/free: moved to backend_tape/nn/recurrent/pair_helpers.c (Phase 1c.7.b). */

/* tensor_gru_cell: moved to backend_tape/nn/recurrent/gru_cell.c (Phase 1c.7.c). */

/* Parameter Registry and the DEBUG_PARAM_GRADS / DEBUG_LSTM_TRAJ
   diagnostics: moved to backend_tape/training/param_registry.c
   (Phase 1e.5). */

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    Tensor* t = (Tensor*)h;
    if (t->dtype_tag == DT_F32) {
        float vf = (float)val;
        for (int i = 0; i < t->numel; i++) ((float*)t->data)[i] -= vf;
    } else {
        for (int i = 0; i < t->numel; i++) ((double*)t->data)[i] -= val;
    }
    return h;
}

/* ================================================================
   Convenience functions
   ================================================================ */

/* Create a one-hot encoded 1D tensor from token indices.
   tokens: array of token indices (int), n_tokens long
   vocab_size: number of classes per token
   Output: 1D tensor of length n_tokens * vocab_size */
TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
    (void)dtag;  /* one-hot's 0/1 image fits losslessly in every dtype; today
                    the live callers (Mnist / Gpt / Transformer) hit the F64
                    path, so the dtag is currently a no-op here — when a
                    non-F64 caller appears, route through tape_round_to_dtype
                    + the dtag-keyed arena allocators added in the parity
                    work. */
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

/* tensor_alloc_doubles / tensor_free_doubles / tensor_read_double /
 * tensor_ptr_array_alloc live in shared_utils.c (unified across all
 * backends; see packages/backends/shared_utils.{c,h}). */

/* tensor_stack_from_array + tensor_cat_from_array: moved to
   backend_tape/linear/concat/stack.c (Phase 1d.2.f). */

/* tensor_create_param_{1,2,3,4}d and tensor_create_state_{1,2}d:
   moved to backend_tape/training/param_create.c (Phase 1e.3). */

/* ================================================================
   Per-dtype creation variants (legacy, pre-unified-dispatch)
   --------------------------------------------------------------
   These per-dtype `_f32` / `_f64` symbols predate the unified
   `tensor_create_<shape>_streamed(..., int dtag)` entry points that
   landed with the FFI tag-dispatch unification (2026-05-22). The
   typed Idris surface now routes through the unified streamed
   symbols exclusively; these per-dtype variants are kept only for
   `backend.h` ABI completeness across the multi-link dylib (every
   backend must export every prototype declared in backend.h).

   The _f64 variants delegate to the unsuffixed F64 creators —
   identity, F64 is the lingua-franca path. The _f32 variants are
   left as abort stubs here even though tape now has *real* F32
   storage via `tape_arena_f32_from_doubles` /
   `tape_persistent_f32_from_doubles` (used by the unified streamed
   path); no caller reaches the legacy `_f32` symbols any more
   (`grep tensor_create_.*_f32` in `packages/idris-ml*` /
   `packages/idris-ml-examples` finds zero hits), so the abort
   diagnostic is reachable only via direct C linkage. Cleaning them
   out is a separate ABI rewrite (paired with row 21 RuntimeDType
   reorder); the abort gives a clear diagnostic if anything ever
   does.
   ================================================================ */

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

/* Per-dtype cast primitives (legacy, pre-unified-dispatch).
 * The live cast path is `tensor_cast_dtype_streamed(src, stream_tag, dtag)`
 * which handles every dtag with the matching real storage (F32 → real
 * `float*` arena; F64 identity; the 8 inference dtypes via
 * `tape_retag_round` + the lingua franca). These per-dtype `_f64` / `_f32`
 * symbols are kept only for backend.h ABI completeness in the multi-link
 * dylib; the typed Idris surface no longer reaches them. F64 is an alias
 * (the FFI wrapper retains the handle and Idris gets a fresh wrapper
 * around the same C handle; gradients flow through the source's tape entry
 * — no new tape op is appended since the operation is observationally
 * identity). F32 aborts here; reach the real F32 cast path via
 * `tensor_cast_dtype_streamed(src, _, 0)`. */
TensorHandle tensor_cast_dtype_f64(TensorHandle src)                                     { return src; }
TensorHandle tensor_cast_dtype_f32(TensorHandle src)                                     { (void)src; return tape_f32_unsupported("tensor_cast_dtype_f32"); }

/* tensor_view_2d, tensor_view_1d: moved to backend_tape/linear/shape/ (Phase 1b.1.d). */

double tensor_item_2d(TensorHandle h, int row, int col) {
    Tensor* t = (Tensor*)h;
    return tape_load_d(t, row * t->shape[1] + col);
}

/* tensor_item_1d: moved to backend_tape/core/lifecycle/item1d.c (Phase 1a.10). */

/* ================================================================
   Native Optimizer + clip_grad_*_opt + serialization accessors +
   native_train_step / optimizer_step_with_clip:
   moved to backend_tape/training/optimizer.c (Phase 1e.6).
   ================================================================ */


/* ================================================================
   System
   ================================================================ */

/* get_rss_mb / get_current_rss_mb live in shared_utils.c (compiled
 * once, unified symbol). Local callers in this file resolve them
 * via the unsuffixed names because both symbols are in the rename
 * header's EXCLUDE set. */

void backend_reset_for_eval(void) {
    tape_reset();
    /* Re-register params so they have valid tape indices */
    for (int j = 0; j < param_count(); j++) {
        Tensor* t = (Tensor*)param_tensor(j);
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

int tensor_live_count(int dummy) { (void)dummy; return (int)tape_size; }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_tape_peak; }

/* backend_memory_report and backend_supports_tensor_params removed
 * (no Idris-side callers). */

/* backend_epoch_begin / backend_profile_reset / backend_profile_report:
   moved to backend_tape/training/profiling.c (Phase 1e.7). */


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
        printf("%.6f\n", ((double*)t->data)[0]);
    } else {
        printf("[");
        for (int i = 0; i < t->numel; i++) {
            if (i > 0) printf(", ");
            printf("%.6f", ((double*)t->data)[i]);
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

/* tensor_write_double_return / tensor_ptr_array_set_return /
 * tensor_alloc_ints / tensor_free_ints / tensor_write_int_return
 * live in shared_utils.c. */

double* tensor_to_doubles_return(TensorHandle h, double* buf) {
    tensor_to_doubles(h, buf);
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

/* native_train_step + optimizer_step_with_clip: moved to backend_tape/training/optimizer.c (Phase 1e.6). */


void* idrisml_seq(void* a, void* b) {
    (void)a;
    return b;
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

/* dropout_random_seed lives in shared_utils.c. */



/* ---- Unified dtag-dispatch create/cast entry points ----
   One symbol per shape, dtag-keyed, superseding the per-dtype
   *_f32_streamed / *_f64_streamed wrappers. dtag 1 (F64) is the hot path —
   plain F64 create, untouched. dtag 0 (F32) allocates real float storage
   so the F32 elementwise kernels can read `(float*)t->data` directly.
   Every other dtag (BF16, F16, I8/I16/I32/I64, U8, Bool) is inference-only: builds the
   tensor as F64 then rounds each value through the target dtype, leaving
   the storage as a double buffer (Phase 2 `tape_retag_round` lingua
   franca). The Idris `Compatible` gate still keeps tape at F64 + F32
   trainable; the inference dtypes wait for Phase 4 to open. */
static TensorHandle tape_retag_round(TensorHandle h, int dtag) {
    Tensor* t = (Tensor*)h;
    int tag = tape_tag_from_dtag(dtag);
    for (int i = 0; i < t->numel; i++)
        ((double*)t->data)[i] = tape_round_to_dtype(((double*)t->data)[i], tag);
    t->dtype_tag = tag;
    return h;
}

/* Build a real-F32-storage arena tensor from F64 source data; the source
   buffer is freed (matches the F64 streamed-create convention where the
   underlying *_f64 creator owns + frees its `data` argument). */
static TensorHandle tape_arena_f32_from_doubles(int* shape, int rank,
                                                double* data, int rg) {
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    float* arena_d = arena_alloc(numel * sizeof(float));
    for (int i = 0; i < numel; i++) arena_d[i] = (float)data[i];
    free(data);
    return make_tensor_arena_f32(arena_d, numel, shape, rank, rg);
}

/* Same, but persistent (malloc'd) — for params / state. tape_append is
   called only when requires_grad; mirrors tensor_create_param_*_f64. */
static TensorHandle tape_persistent_f32_from_doubles(int* shape, int rank,
                                                     double* data, int rg) {
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(float));
    for (int i = 0; i < numel; i++) ((float*)t->data)[i] = (float)data[i];
    free(data);
    t->shape = malloc(rank * sizeof(int));
    memcpy(t->shape, shape, rank * sizeof(int));
    t->rank = rank;
    t->numel = numel;
    t->requires_grad = rg;
    t->tape_idx = -1;
    t->persistent = 1;
    t->dtype_tag = DT_F32;
    if (rg) tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

/* Under the kind-major dtag layout (closed 2026-05-23): dtag 15 = F64 (the
   lingua franca, fast-path to the f64 creator), dtag 14 = F32 (real 4-byte
   float storage). All other valid dtags (Bool=1, U8=4, I8/I16/I32/I64,
   F16=13, BF16=17) route through tape_retag_round's lingua-franca path
   (store doubles in F64 layout, retag to the inference dtype). Invalid
   dtags abort via tape_tag_from_dtag's default arm. */
TensorHandle tensor_create_scalar_streamed(double value, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_scalar_f64(value, requires_grad);
    if (dtag == 14) return make_scalar_f32(value, requires_grad);
    return tape_retag_round(tensor_create_scalar_f64(value, requires_grad), dtag);
}
TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_f64(data, shape, rank, requires_grad);
    if (dtag == 14) {
        /* tensor_create_f64 copies + then frees the input; mirror that contract.
           We need our own free since we bypass the f64 creator. */
        int numel = 1;
        for (int i = 0; i < rank; i++) numel *= shape[i];
        float* arena_d = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) arena_d[i] = (float)data[i];
        return make_tensor_arena_f32(arena_d, numel, shape, rank, requires_grad);
    }
    return tape_retag_round(tensor_create_f64(data, shape, rank, requires_grad), dtag);
}
TensorHandle tensor_create_1d_streamed(int n, double* data, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_1d_f64(n, data, requires_grad);
    if (dtag == 14) { int s[] = {n}; return tape_arena_f32_from_doubles(s, 1, data, requires_grad); }
    return tape_retag_round(tensor_create_1d_f64(n, data, requires_grad), dtag);
}
TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_2d_f64(rows, cols, data, requires_grad);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_arena_f32_from_doubles(s, 2, data, requires_grad); }
    return tape_retag_round(tensor_create_2d_f64(rows, cols, data, requires_grad), dtag);
}
TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_1d_f64(n, data);
    if (dtag == 14) { int s[] = {n}; return tape_persistent_f32_from_doubles(s, 1, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_1d_f64(n, data), dtag);
}
TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_2d_f64(rows, cols, data);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_persistent_f32_from_doubles(s, 2, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_2d_f64(rows, cols, data), dtag);
}
TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_3d_f64(d0, d1, d2, data);
    if (dtag == 14) { int s[] = {d0, d1, d2}; return tape_persistent_f32_from_doubles(s, 3, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_3d_f64(d0, d1, d2, data), dtag);
}
TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
    if (dtag == 14) { int s[] = {d0, d1, d2, d3}; return tape_persistent_f32_from_doubles(s, 4, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_4d_f64(d0, d1, d2, d3, data), dtag);
}
TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_state_1d_f64(n, data);
    if (dtag == 14) { int s[] = {n}; return tape_persistent_f32_from_doubles(s, 1, data, /*rg=*/0); }
    return tape_retag_round(tensor_create_state_1d_f64(n, data), dtag);
}
TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_state_2d_f64(rows, cols, data);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_persistent_f32_from_doubles(s, 2, data, /*rg=*/0); }
    return tape_retag_round(tensor_create_state_2d_f64(rows, cols, data), dtag);
}
TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag) {
    (void)stream_tag;
    Tensor* s = (Tensor*)src;
    int tag = tape_tag_from_dtag(dtag);
    /* F64 → F64 stays observational identity (preserves autograd-through-cast).
       The shortcut applies only when the *source* is already F64 — casting a
       non-F64 source up to F64 must still produce a fresh F64-tagged tensor. */
    if (tag == DT_F64 && s->dtype_tag == DT_F64) return tensor_cast_dtype_f64(src);
    /* F32 target: real F32 storage (4 bytes/elem), matching the Phase 3
       streamed-create path. Skipping this and falling through to
       tape_retag_round would produce a lingua-franca F32 (double storage,
       DT_F32 tag) — internally consistent for tensor_item_1d (reads as
       double*) but garbage for tensor_to_doubles / tape_load_d / the F32
       kernels (all assume 4-byte-per-elem float storage). tape_load_d on
       the read side normalizes real-F32 vs lingua-franca sources. */
    if (tag == DT_F32) {
        if (s->rank == 0) {
            double v = tape_load_d(s, 0);
            return make_scalar_f32(v, 0);
        }
        float* arena_d = arena_alloc(s->numel * sizeof(float));
        for (int i = 0; i < s->numel; i++) arena_d[i] = (float)tape_load_d(s, i);
        return make_tensor_arena_f32(arena_d, s->numel, s->shape, s->rank, 0);
    }
    /* Otherwise: a fresh non-grad tensor cloning src's values, rounded into the
       target dtype. (Inference / precision-demo casts are NoGrad.) Source is
       read via tape_load_d so a real-F32 source casts correctly into the
       lingua-franca target. */
    if (s->rank == 0) {
        double v = tape_load_d(s, 0);
        return tape_retag_round(make_scalar(v, 0), dtag);
    }
    double* arena_d = arena_alloc(s->numel * sizeof(double));
    for (int i = 0; i < s->numel; i++) arena_d[i] = tape_load_d(s, i);
    Tensor* t = make_tensor_arena(arena_d, s->numel, s->shape, s->rank, 0);
    return tape_retag_round((TensorHandle)t, dtag);
}

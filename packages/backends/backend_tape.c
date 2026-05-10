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

TensorHandle tensor_cross_entropy(TensorHandle hinput, TensorHandle htarget) {
    /* Simplified: compute -sum(target * log_softmax(input)) / n */
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_cross_entropy");
    TensorHandle ls = tensor_log_softmax(hinput, 0);
    Tensor* lsT = (Tensor*)ls;
    double loss = 0;
    for (int i = 0; i < lsT->numel; i++) loss -= tape_load_d(target, i) * tape_load_d(lsT, i);
    loss /= lsT->numel;
    return (input->dtype_tag == DT_F32) ? make_scalar_f32(loss, 0) : make_scalar(loss, 0);
}

TensorHandle tensor_mse_loss(TensorHandle hinput, TensorHandle htarget) {
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_mse_loss");
    double loss = 0;
    for (int i = 0; i < input->numel; i++) {
        double d = tape_load_d(input, i) - tape_load_d(target, i);
        loss += d * d;
    }
    double mean = loss / input->numel;
    return (input->dtype_tag == DT_F32) ? make_scalar_f32(mean, 0) : make_scalar(mean, 0);
}

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

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    Tensor* input = (Tensor*)hinput;
    Tensor* gamma = (Tensor*)hgamma;
    Tensor* beta = (Tensor*)hbeta;
    int n = channels * spatial;
    int chPerGroup = channels / numGroups;
    int groupSize = chPerGroup * spatial;

    if (input->dtype_tag != gamma->dtype_tag || input->dtype_tag != beta->dtype_tag)
        tape_abort_mixed_dtype("tensor_group_norm");
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {n};
    void* out = is_f32 ? (void*)arena_alloc(n * sizeof(float))
                       : (void*)calloc(n, sizeof(double));
    for (int g = 0; g < numGroups; g++) {
        double mean = 0;
        int base = g * groupSize;
        for (int j = 0; j < groupSize; j++) mean += tape_load_d(input, base + j);
        mean /= groupSize;
        double var = 0;
        for (int j = 0; j < groupSize; j++) {
            double d = tape_load_d(input, base + j) - mean;
            var += d * d;
        }
        var /= groupSize;
        double rstd = 1.0 / sqrt(var + eps);
        for (int c = 0; c < chPerGroup; c++) {
            int absC = g * chPerGroup + c;
            double gc = tape_load_d(gamma, absC);
            double bc = tape_load_d(beta, absC);
            for (int s = 0; s < spatial; s++) {
                int idx = absC * spatial + s;
                double x_hat = (tape_load_d(input, idx) - mean) * rstd;
                double v = gc * x_hat + bc;
                if (is_f32) ((float*)out)[idx] = (float)v;
                else        ((double*)out)[idx] = v;
            }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, n, out_shape, 1, input->requires_grad || gamma->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 1, input->requires_grad || gamma->requires_grad); free(out); }
    /* No backward tape entry for now — torch/MLX handle it natively */
    return r;
}

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
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv_transpose1d");
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[1], kL = kernel->shape[2];
    int oL = (L - 1) * stride - 2 * pad + kL;
    int is_f32 = (input->dtype_tag == DT_F32);
    int numel = outC * oL;
    int out_shape[] = {outC, oL};
    int rg = input->requires_grad || kernel->requires_grad;
    /* Compute in double for sum stability; narrow to float on store. */
    double* dbl = calloc(numel, sizeof(double));
    if (bias) for (int oc = 0; oc < outC; oc++)
        for (int ol = 0; ol < oL; ol++) dbl[oc*oL + ol] = tape_load_d(bias, oc);
    for (int ic = 0; ic < inC; ic++)
        for (int il = 0; il < L; il++)
            for (int oc = 0; oc < outC; oc++)
                for (int kl = 0; kl < kL; kl++) {
                    int ol = il * stride - pad + kl;
                    if (ol >= 0 && ol < oL)
                        dbl[oc*oL + ol] += tape_load_d(input, ic*L + il)
                                         * tape_load_d(kernel, ic*outC*kL + oc*kL + kl);
                }
    Tensor* r;
    if (is_f32) {
        float* out = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) out[i] = (float)dbl[i];
        free(dbl);
        r = make_tensor_arena_f32(out, numel, out_shape, 2, rg);
    } else {
        r = make_tensor(dbl, out_shape, 2, rg);
        free(dbl);
    }
    return r;
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int padH, int padW,
                                     int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv_transpose2d");
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[1], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H - 1) * strideH - 2 * padH + kH;
    int oW = (W - 1) * strideW - 2 * padW + kW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int numel = outC * oH * oW;
    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad;
    double* dbl = calloc(numel, sizeof(double));
    if (bias) for (int oc = 0; oc < outC; oc++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) dbl[oc*oH*oW + oh*oW + ow] = tape_load_d(bias, oc);
    for (int ic = 0; ic < inC; ic++)
        for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
                for (int oc = 0; oc < outC; oc++)
                    for (int kh = 0; kh < kH; kh++)
                        for (int kw = 0; kw < kW; kw++) {
                            int oh = ih*strideH - padH + kh;
                            int ow = iw*strideW - padW + kw;
                            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
                                dbl[oc*oH*oW + oh*oW + ow] += tape_load_d(input, ic*H*W + ih*W + iw)
                                    * tape_load_d(kernel, ic*outC*kH*kW + oc*kH*kW + kh*kW + kw);
                        }
    Tensor* r;
    if (is_f32) {
        float* out = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) out[i] = (float)dbl[i];
        free(dbl);
        r = make_tensor_arena_f32(out, numel, out_shape, 3, rg);
    } else {
        r = make_tensor(dbl, out_shape, 3, rg);
        free(dbl);
    }
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
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv1d_grouped");
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[0];
    int inC_g = inC / groups;
    int outC_g = outC / groups;
    int kL = kernel->shape[2];
    int oL = (L + 2*pad - kL) / stride + 1;
    int total = outC * oL;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {outC, oL};
    int rg = input->requires_grad || kernel->requires_grad;
    void* out = is_f32 ? (void*)arena_alloc(total * sizeof(float))
                       : (void*)calloc(total, sizeof(double));
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < outC_g; oc++) {
            int abs_oc = g * outC_g + oc;
            for (int ol = 0; ol < oL; ol++) {
                double val = bias ? tape_load_d(bias, abs_oc) : 0.0;
                for (int ic = 0; ic < inC_g; ic++) {
                    int abs_ic = g * inC_g + ic;
                    for (int kl = 0; kl < kL; kl++) {
                        int il = ol * stride - pad + kl;
                        if (il >= 0 && il < L)
                            val += tape_load_d(input, abs_ic*L + il)
                                 * tape_load_d(kernel, abs_oc*inC_g*kL + ic*kL + kl);
                    }
                }
                if (is_f32) ((float*)out)[abs_oc*oL + ol] = (float)val;
                else        ((double*)out)[abs_oc*oL + ol] = val;
            }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, total, out_shape, 2, rg);
    else { r = make_tensor((double*)out, out_shape, 2, rg); free(out); }
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
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv2d_grouped");
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[0];
    int inC_g = inC / groups;
    int outC_g = outC / groups;
    int kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int numel = outC * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad;
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < outC_g; oc++) {
            int abs_oc = g * outC_g + oc;
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double val = bias ? tape_load_d(bias, abs_oc) : 0.0;
                    for (int ic = 0; ic < inC_g; ic++) {
                        int abs_ic = g * inC_g + ic;
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh*strideH - padH + kh;
                                int iw = ow*strideW - padW + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    val += tape_load_d(input, abs_ic*H*W + ih*W + iw)
                                         * tape_load_d(kernel, abs_oc*inC_g*kH*kW + ic*kH*kW + kh*kW + kw);
                            }
                    }
                    if (is_f32) ((float*)out)[abs_oc*oH*oW + oh*oW + ow] = (float)val;
                    else        ((double*)out)[abs_oc*oH*oW + oh*oW + ow] = val;
                }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 3, rg);
    else { r = make_tensor((double*)out, out_shape, 3, rg); free(out); }
    return r;
}

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

/* Forward declarations for profiling */
#include <sys/time.h>
double _wall_ms(void) {  /* non-static so tape.c can extern it */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
double prof_forward_ms = 0, prof_backward_ms = 0, prof_optimizer_ms = 0;
int prof_forward_ops = 0, prof_backward_ops = 0, prof_epochs = 0;
double prof_epoch_start = 0; /* set by backend_epoch_begin() */
int prof_backward_processed = 0, prof_backward_skipped = 0;
double prof_backward_per_op[OP_COUNT] = {0};
int prof_backward_count_per_op[OP_COUNT] = {0};
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
    return strncmp(param_name(i), opt->prefix, strlen(opt->prefix)) == 0;
}

/* Compute total number of elements across all params (for per-element optimizer buffers) */
static int param_total_elements(void) {
    int total = 0;
    for (int i = 0; i < param_count(); i++)
        total += ((Tensor*)param_tensor(i))->numel;
    return total;
}

/* Offset into the flat per-element buffer for param i, element j */
static int param_element_offset(int param_idx) {
    int off = 0;
    for (int i = 0; i < param_idx; i++)
        off += ((Tensor*)param_tensor(i))->numel;
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
    for (int i = 0; i < param_count(); i++) {
        const char* on_name = param_name(i);
        if (strncmp(on_name, online_scope, on_len) != 0) continue;
        /* Build target name: target_scope ++ (on_name + on_len). */
        char tgt_name[256];
        size_t suffix_len = strlen(on_name + on_len);
        if (tg_len + suffix_len + 1 > sizeof(tgt_name)) continue;
        memcpy(tgt_name, target_scope, tg_len);
        memcpy(tgt_name + tg_len, on_name + on_len, suffix_len + 1);
        /* Find target param. */
        for (int j = 0; j < param_count(); j++) {
            if (strcmp(param_name(j), tgt_name) != 0) continue;
            Tensor* on_t = (Tensor*)param_tensor(i);
            Tensor* tg_t = (Tensor*)param_tensor(j);
            if (on_t->numel != tg_t->numel) break;
            for (int k = 0; k < on_t->numel; k++) {
                ((double*)tg_t->data)[k] = one_minus_tau * ((double*)tg_t->data)[k] + tau * ((double*)on_t->data)[k];
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
    if (opt->param_lr == NULL || opt->param_lr_count < param_count()) {
        int new_count = param_count() > 64 ? param_count() : 64;
        double* new_lr = realloc(opt->param_lr, new_count * sizeof(double));
        /* Initialize new entries to -1 (sentinel: use base LR) */
        for (int i = opt->param_lr_count; i < new_count; i++) new_lr[i] = -1.0;
        opt->param_lr = new_lr;
        opt->param_lr_count = new_count;
    }
    /* Find param by name and set its LR */
    for (int i = 0; i < param_count(); i++) {
        if (strcmp(param_name(i), name) == 0) {
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

    for (int i = 0; i < param_count(); i++) {
        if (!opt_owns_param(opt, i)) continue;
        /* Phase 1.5e DIAGNOSTIC: SKIP_LSTM_INIT skips updating params whose
           names end in _h0 or _c0 (LSTM learned initial state). Equivalent
           to keeping them as zero state tensors. Used to localize whether
           the convergence regression is in the gradient values being
           applied to h0/c0 vs being elsewhere. Remove once diagnosed. */
        if (getenv("SKIP_LSTM_INIT")) {
            const char* nm = param_name(i);
            size_t L = strlen(nm);
            if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 ||
                           strcmp(nm + L - 3, "_c0") == 0)) {
                continue;
            }
        }
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        int base = param_element_offset(i);

        /* Per-param LR: use override if set, otherwise base LR */
        double lr = opt->lr;
        if (opt->param_lr && i < opt->param_lr_count && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];

        for (int j = 0; j < t->numel; j++) {
            double g = ((double*)t->grad)[j];
            int idx = base + j;  /* per-element index into optimizer buffers */

            /* Dtype-aware reads + writes so F32 params take f32-precision
               updates (asserted by the rung-4 F32-exactness check). Moment
               buffers (opt->m / opt->v) stay F64 — standard mixed-precision
               practice and lets the F64 numerics path stay byte-identical. */
            double w = tape_load_d(t, j);
            switch (opt->type) {
            case 0: /* SGD */
                tape_store_d(t, j, w - lr * g);
                break;

            case 1: { /* RMSprop — keep lr OUTSIDE the momentum buffer to match
                         torch.optim.RMSprop. Folding lr into the buffer
                         (buf = m*buf + lr*g/avg) coincides with PyTorch only at
                         constant lr; under an LR schedule the buffer carries
                         stale rates and diverges. */
                opt->v[idx] = opt->alpha * opt->v[idx] + (1.0 - opt->alpha) * g * g;
                double avg = sqrt(opt->v[idx]) + opt->eps;
                if (opt->momentum > 0) {
                    opt->m[idx] = opt->momentum * opt->m[idx] + g / avg;
                    tape_store_d(t, j, w - lr * opt->m[idx]);
                } else {
                    tape_store_d(t, j, w - lr * g / avg);
                }
                break;
            }

            case 2: { /* Adam */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                tape_store_d(t, j, w - lr * mhat / (sqrt(vhat) + opt->eps));
                break;
            }

            case 3: { /* AdamW (decoupled weight decay) */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                double w1 = w - lr * mhat / (sqrt(vhat) + opt->eps);
                tape_store_d(t, j, w1 - lr * opt->weight_decay * w1);
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
    for (int j = 0; j < param_count(); j++) {
        Tensor* t = (Tensor*)param_tensor(j);
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
    for (int i = 0; i < param_count(); i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) {
            if (((double*)t->grad)[j] > max_val) ((double*)t->grad)[j] = max_val;
            if (((double*)t->grad)[j] < -max_val) ((double*)t->grad)[j] = -max_val;
        }
    }
}

/* Internal clip-norm helper scoped to params owned by `opt`. */
static double clip_grad_norm_opt(Optimizer* opt, double max_norm) {
    double total = 0;
    for (int i = 0; i < param_count(); i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) total += ((double*)t->grad)[j] * ((double*)t->grad)[j];
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count(); i++) {
            if (opt && !opt_owns_param(opt, i)) continue;
            Tensor* t = (Tensor*)param_tensor(i);
            if (!t->grad) continue;
            for (int j = 0; j < t->numel; j++) ((double*)t->grad)[j] *= scale;
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
    return param_count();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, ((Tensor*)param_tensor(idx))->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(out, opt->m + offset, numel * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, ((Tensor*)param_tensor(idx))->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(out, opt->v + offset, numel * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(opt->m + offset, data, numel * sizeof(double));
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
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
            param_count(), ({int n=0; for(int i=0;i<param_count();i++) n+=((Tensor*)param_tensor(i))->numel; n;}));
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
